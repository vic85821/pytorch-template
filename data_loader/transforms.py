import scipy.ndimage
import random
import SimpleITK as sitk
import numpy as np

class Normalization(object):
    """Normalization: apply Z-normalization to the volume.
    """

    def __init__(self):        
        pass

    def __call__(self, data):
        image, label = data['image'], data['label']
        image = (image - np.mean(image)) / np.std(image)
        return {'image': image, 'label': label}

class RandomElasticDeformation(object):
    """RandomElasticDeformation: data augmentation using elastic deformations as used by V-Net. 
        ref: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/rand_elastic_deform.html#RandomElasticDeformationLayer
    """
    
    def __init__(self, num_controlpoints=4, std_deformation_sigma=15, proportion_to_augment=0.5):
        self.num_controlpoints = max(num_controlpoints, 2)
        self.std_deformation_sigma = max(std_deformation_sigma, 1)
        self.proportion_to_augment = proportion_to_augment
        self.bspline_transformation = None
    
    def randomise_bspline_transformation(self, shape):
        itkimg = sitk.GetImageFromArray(np.zeros(shape))
        trans_from_domain_mesh_size = [self.num_controlpoints] * itkimg.GetDimension()
        self.bspline_transformation = sitk.BSplineTransformInitializer(itkimg, trans_from_domain_mesh_size)

        params = self.bspline_transformation.GetParameters()
        params_numpy = np.asarray(params, dtype=float)
        params_numpy = params_numpy + np.random.randn(params_numpy.shape[0]) * self.std_deformation_sigma
        
        # remove z deformations! The resolution in z is too bad
        params_numpy[0:int(len(params) / 3)] = 0
        
        params = tuple(params_numpy)
        self.bspline_transformation.SetParameters(params)
    
    def apply_bspline_transformation(self, image, interp_order=3):
        resampler = sitk.ResampleImageFilter()
        if interp_order > 1:
            resampler.SetInterpolator(sitk.sitkBSpline)
        elif interp_order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interp_order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            return image

        squeezed_image = np.squeeze(image)
        sitk_image = sitk.GetImageFromArray(squeezed_image)
        
        resampler.SetReferenceImage(sitk_image)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(self.bspline_transformation)
        out_img_sitk = resampler.Execute(sitk_image)
        out_img = sitk.GetArrayFromImage(out_img_sitk)
        return out_img.reshape(image.shape)
    
    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.proportion_to_augment:
            if image.ndim == 3:
                self.randomise_bspline_transformation(image.shape)
                image = self.apply_bspline_transformation(image, interp_order=3)
                label = self.apply_bspline_transformation(label, interp_order=0)            
            elif image.ndim == 4:
                self.randomise_bspline_transformation(image.shape[:-1])
                image = np.stack([self.apply_bspline_transformation(image[..., i], interp_order=3) for i in range(image.shape[-1])], axis=-1)
                label = self.apply_bspline_transformation(label, interp_order=0)
        return {'image': image, 'label': label}
        
class RandomCrop(object):
    """RandomCrop: randomly crop the volume and its corresponding label.
    Args:
        output_size (tuple): desired output size.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        image, label = data['image'], data['label']
        if image.ndim == 3:
            h, w, d = image.shape
            h_new, w_new, d_new = self.output_size
            h0, w0, d0 = random.randint(0, h - h_new), random.randint(0, w - w_new), random.randint(0, d - d_new)
            image = image[h0 : h0 + h_new, w0 : w0 + w_new, d0 : d0 + d_new]
            label = label[h0 : h0 + h_new, w0 : w0 + w_new, d0 : d0 + d_new]
        elif image.ndim == 4:
            h, w, d = image.shape[:-1]
            h_new, w_new, d_new = self.output_size
            h0, w0, d0 = random.randint(0, h - h_new), random.randint(0, w - w_new), random.randint(0, d - d_new)
            image = np.stack([image[..., i][h0 : h0 + h_new, w0 : w0 + w_new, d0 : d0 + d_new] for i in range(image.shape[-1])], axis=-1)
            label = label[h0 : h0 + h_new, w0 : w0 + w_new, d0 : d0 + d_new]
        return {'image': image, 'label': label}