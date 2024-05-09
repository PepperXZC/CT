import astra, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import  xml.dom.minidom

root = '/data1/xzc/0423'
obj = '423_2_left'

root = os.path.join(root, obj)
datart = os.path.join(root, 'ScanData')
refrt = os.path.join(root, 'ScanRef')
calib = os.path.join(root, 'calibration.xml')
info = os.path.join(root, 'info.xml')

datafs = os.listdir(datart)
data = np.zeros(shape=(len(datafs), 480, 640), dtype=np.uint16)
x_vol_sz, y_vol_sz, z_vol_sz = 640, 640, 480 # volex number for x, y, z. Please make sure x_vol_sz=y_vol_sz
num_angles = len(datafs)
for file in datafs:
    if not os.path.isdir(file):
        p = os.path.join(datart, file)
        f = Image.open(p)
        L = f.convert('L')
        num = int(p.split('/')[-1][:3])
        parr = np.array(L).astype(np.uint16)
        data[num-1] = parr

angles = np.linspace(0, 2*np.pi, num_angles)
# print(data.shape)

reffs = os.listdir(refrt)
ref = np.zeros(shape=(len(reffs), 480, 640), dtype=np.uint16)
num_angles = len(reffs)
for file in reffs:
    if not os.path.isdir(file):
        p = os.path.join(refrt, file)
        f = Image.open(p)
        L = f.convert('L')
        num = int(p.split('/')[-1][:3])
        parr = np.array(L).astype(np.uint16)
        ref[num-1] = parr


#打开xml文档
dom = xml.dom.minidom.parse(calib)
root = dom.documentElement
calibkeys = ['SourceToAxis','AxisToDetector', 'HorizLightSize',
             'VertLightSize', 'AxisOfRotationOffset', 
             'EquatorialOffset', 'HorizPixelSize','VertPixelSize']
calibdic = {}
for key in calibkeys:
    bb = root.getElementsByTagName(key)
    calibdic[key] = float(bb[0].firstChild.data)

dom = xml.dom.minidom.parse(info)
root = dom.documentElement
infokeys = ['FrameRate', 'ShutterSpeed', 'ReferenceProjections', 'DataProjections']
infodic = {}
for key in infokeys:
    bb = root.getElementsByTagName(key)
    infodic[key] = float(bb[0].firstChild.data)

# num_angles, num_detectors_orig = sinogram.shape

proj_type = 'cuda3d'
cm_per_pixel = calibdic['HorizPixelSize']

# source_origin_cm = source_origin_cm # distance b/n source and COR  [cm]
source_origin_mm = calibdic['SourceToAxis'] * 10 # distance b/n source and COR  [mm]
# detector_origin_cm = detector_origin_cm # distance b/n COR and detector[cm]
detector_origin_mm = calibdic['AxisToDetector'] * 10 # distance b/n COR and detector[mm]

# source_origin = source_origin_cm / cm_per_pixel # Source-COR distance [pixels]
# detector_origin = detector_origin_cm / cm_per_pixel # COR-detector distance [pixels]

# magnification = (source_origin + detector_origin) / source_origin
# detector_pixel_size = magnification
detector_pixel_size_x = calibdic['HorizLightSize'] * 10 / 480
detector_pixel_size_y = calibdic['VertLightSize'] * 10 / 640

# vecs = cal_vecs(src_x_det_crd, src_y_det_crd, src_z_det_crd, rot_x_det_crd, rot_y_det_crd, rot_z_det_crd, angles, det_spacing_x, det_spacing_y)
proj_geom = astra.create_proj_geom('cone',
                                    detector_pixel_size_x,
                                    detector_pixel_size_y,
                                    480,
                                    640,
                                    angles,
                                    source_origin_mm,
                                    detector_origin_mm)

# FOV_x = calibdic['HorizLightSize'] * source_origin_mm / (source_origin_mm+detector_origin_mm)
FOV_y = calibdic['VertLightSize']* 10 * source_origin_mm / (source_origin_mm+detector_origin_mm)
vol_geom = astra.create_vol_geom(640, 640, 640, 
                                 (-640/2)*FOV_y/640, (640/2)*FOV_y/640,
                                 (-640/2)*FOV_y/640, (640/2)*FOV_y/640,
                                 (-640/2)*FOV_y/640, (640/2)*FOV_y/640,)
# Create the actual projector 
proj_id = astra.create_projector(proj_type, 
                                proj_geom, 
                                vol_geom)

# source_origin_cm = source_origin_cm
# detector_origin_cm = detector_origin_cm

ops = np.log(ref / data)
ops[np.isnan(ops)] = 0
ops = ops.astype(np.float32)
ops = np.transpose(ops, (1,0,2))

plt.subplot(131)1
plt.imshow(ops[:, :, 200], cmap='gray')
# plt.colorbar()
plt.subplot(132)
plt.imshow(ops[:,400,:], cmap='gray')
# plt.colorbar()
plt.subplot(133)
plt.imshow(ops[400,:,:], cmap='gray')
# plt.colorbar()
plt.savefig('ct.png')
print(ops.shape)
vol_sz = [x_vol_sz, y_vol_sz, z_vol_sz] # CStalk
vol_rec = np.zeros(vol_sz, dtype=np.float32)

# astra.create_sino(V_exact, proj_id)
sino_id = astra.data3d.create('-sino', proj_geom, ops)

recon_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict('FDK_CUDA')
# cfg['ProjectorId'] = proj_id
cfg['ProjectionDataId'] = sino_id
cfg['ReconstructionDataId'] = recon_id
fbp_id = astra.algorithm.create(cfg)
astra.algorithm.run(fbp_id)
V = astra.data3d.get(recon_id)
pro = astra.data3d.get(sino_id)
astra.algorithm.delete(fbp_id)
astra.data3d.delete(recon_id)
astra.data3d.delete(proj_id)
plt.figure()
plt.subplot(121)
plt.gray()
plt.imshow(V[500,:,:])
plt.colorbar(shrink=0.5)
plt.subplot(122)
plt.gray()
plt.imshow(V[:,300, :])
plt.colorbar(shrink=0.5)
plt.savefig('ctrecon.png')
plt.show()
