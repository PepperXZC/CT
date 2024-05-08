import astra, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root = '/data1/xzc/0423'
obj = '423_2_left'

root = os.path.join(root, obj)
datart = os.path.join(root, 'ScanData')
refrt = os.path.join(root, 'ScanRef')

datafs = os.listdir(datart)
data = np.zeros(shape=(len(datafs), 480, 640), dtype=np.uint16)
num_angles = len(datafs)
for file in datafs:
    if not os.path.isdir(file):
        p = os.path.join(datart, file)
        f = Image.open(p)
        L = f.convert('L')
        num = int(p.split('/')[-1][:3])
        parr = np.array(L).astype(np.uint16)
        data[num-1] = parr

angles = np.linspace(0, np.pi, num_angles)
print(data.shape)

def create_projector_id(sinogram,
                        beam_geometry,
                        projector_type = 'cuda', 
                        recon_dimension = None,
                        source_origin_cm = None,
                        detector_origin_cm = None):
        '''
        ASTRA projection geometry entry (https://www.astra-toolbox.com/apiref/creators.html):
            astra.create_proj_geom('parallel', detector_spacing, det_count, angles)
            astra.create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
                
        ASTRA projector entry:
            astra.create_projector(proj_type, 
                                   proj_geom, 
                                   recon_geom)
            proj_type: 
                For CPU operation, use: line or strip or linear 
                For GPU operation, use: cuda
            
            Reconstruction geometry:
                default = (N, N) where N = number of original detectors in projection window
        '''
        proj_type = projector_type
        
        num_detector_pixels = sinogram.shape[1]
        if beam_geometry == 'parallel':
            rel_detector_size = 1.0 # No magnification in parallel beam geometry
            proj_geom = astra.create_proj_geom('parallel', 
                                               rel_detector_size,
                                               num_detector_pixels,
                                               angles)
        elif beam_geometry == 'fanflat':
            
            if self.Image_data_type == 'mcnp':
                cm_per_pixel = self.detector_length / 512 # 512 F5 point dets spreadout on a 20cm image plane
            elif self.Image_data_type == 'experimental':
                cm_per_pixel = self.detector_length / 2048 # 2048 pixels spreadout on a 20cm image plane
            
            if source_origin_cm:
                source_origin_cm = source_origin_cm # distance b/n source and COR  [cm]
                detector_origin_cm = detector_origin_cm # distance b/n COR and detector[cm]

                source_origin = source_origin_cm / cm_per_pixel # Source-COR distance [pixels]
                detector_origin = detector_origin_cm / cm_per_pixel # COR-detector distance [pixels]

                magnification = (source_origin + detector_origin) / source_origin
                detector_pixel_size = magnification
                
                proj_geom = astra.create_proj_geom('fanflat',
                                                   detector_pixel_size,
                                                   num_detector_pixels,
                                                   angles,
                                                   source_origin,
                                                   detector_origin)
            else:
                print("Fanbeam geometry selected but source_origin_distance not given")
            
            
        if not recon_dimension:
            rec_geometry = self.num_detectors_orig
        else:
            rec_geometry = recon_dimension
        
        vol_geom = astra.create_vol_geom(rec_geometry, rec_geometry)
        # Create the actual projector 
        proj_id = astra.create_projector(proj_type, 
                                         proj_geom, 
                                         vol_geom)
            
#        print('proj_id create: ', proj_id)
            
        return proj_id

proj_id = create_projector_id(sinogram,
                            beam_geometry,
                            projector_type,
                            recon_dimension,
                            source_origin_cm,
                            detector_origin_cm)

proj_geom = astra.create_proj_geom('fanflat',
                                    detector_pixel_size,
                                    num_detector_pixels,
                                    self.angles,
                                    source_origin,
                                    detector_origin)
        # print(parr.shape, parr.dtype)
# astra.create_proj_geom()