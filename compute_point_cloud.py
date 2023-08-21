# reconstruction tutorial from open3d
import os
import numpy as np
import open3d as o3d
# from open3d.web_visualizer import draw
# from open3d.visualization.rendering import  OffscreenRenderer
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation
from PIL import Image
import skvideo.io
import cv2
import argparse


DEPTH_WIDTH = 256
DETH_HEIGHT = 192
MAX_DEPTH = 20.0

def _resize_camera_matrix(camera_matrix, scale_x, scale_y):
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]    
    cx = camera_matrix[0,2]    
    cy = camera_matrix[1,2]    
    return np.array([[fx * scale_x, 0.0,cx * scale_x],
                     [0.,fy * scale_y, cy * scale_y],
                     [0.,0.,1.0]])

def _show_images(color, depth):
    plt.subplot(1, 2, 1)
    plt.title('Color')
    plt.imshow(color)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title('Depth')
    plt.imshow(depth)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def get_intrinsics(data):
    intrinsics_scaled = _resize_camera_matrix(data, DEPTH_WIDTH / 1920, DETH_HEIGHT / 1440)
    return o3d.camera.PinholeCameraIntrinsic(width=DEPTH_WIDTH, height=DETH_HEIGHT, fx=intrinsics_scaled[0,0],
                                             fy=intrinsics_scaled[1,1], cx=intrinsics_scaled[0,2], cy=intrinsics_scaled[1,2])

def load_depth(path, confidence=None, filter_level=0):
    if path[-4:] == '.npy':
        depth_mm = np.load(path)
    elif path[-4:] == '.png':
        depth_mm = np.array(Image.open(path))
    depth_m = depth_mm.astype(np.float32) / 1000.0
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    return o3d.geometry.Image(depth_m)

def load_confidence(path):
    return np.array(Image.open(path))
def read_data(path):
    intrinsics = np.loadtxt(os.path.join(path, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(path, 'odometry.csv'), delimiter=',', skiprows=1)
    
    poses = []
    
    for line in odometry:
        position = line[2:5]
        quaternion = line[5:]
        T_WC = np.eye(4)
        T_WC[:3,:3] = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3,3] = position
        poses.append(T_WC)
    
    depth_dir = os.path.join(path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]
    
    return {'poses': poses, 'intrinsics': intrinsics, 'depth_frames': depth_frames, 'path':path}
    
def point_clouds(data):
    pcs = []
    intrinsics = get_intrinsics(data['intrinsics'])
    pc = o3d.geometry.PointCloud()
    meshes = []
    rgb_path = os.path.join(data['path'],'rgb.mp4')
    
    cap = cv2.VideoCapture(rgb_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    frame_index = 0
    while(cap.isOpened()):
        ret, rgb = cap.read()
        if not ret:
            break
        frame_index +=1
        
        if frame_index % 5 != 0:
            continue
        
        T_WC = data['poses'][frame_index]    
        T_CW = np.linalg.inv(T_WC)
        
        
        # confidence
        confidence = load_confidence(os.path.join(data['path'],'confidence',f'{frame_index:06}.png'))
        
        depth_path = data['depth_frames'][frame_index]
        depth = load_depth(depth_path, confidence, filter_level=2)
        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((DEPTH_WIDTH, DETH_HEIGHT))
        rgb = np.array(rgb)
        
        #_show_images(rgb, depth)
        

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), depth,
            depth_scale=1.0 , depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)
        pc += o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsic=T_CW)
        
        #print(rgb.shape, depth)
        
        
    return [pc]


if __name__=='__main__':
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='path to dataset')
    parser.add_argument('--sample_name', type=str, required=True, help='sample name')

    args = parser.parse_args()
    path = os.path.join(args.dataset_path, args.sample_name)
    # check if path exists
    if not os.path.exists(path):
        raise ValueError(f'Path {path} does not exist')
    
    path_colors = os.path.join(path, "color")
    path_depths = os.path.join(path, "depth")

    data = read_data(path)
    pc = point_clouds(data)
    directory_name = os.path.split(path)[-1]
    os.makedirs(args.dataset_path+f'/Area_5/{directory_name}', exist_ok=True)
    o3d.io.write_point_cloud(filename=args.dataset_path+f'/Area_5/{directory_name}/{directory_name}.pts', pointcloud=pc[0], print_progress=True)


