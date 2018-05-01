# coding: utf-8
import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

intensity_thresold = 40

#Function to detect pole or all other line markings
def detect_obj(image, detectOnlyPole):
    # grayscale to the image
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/grayscale_img.jpg', grayscale)
    
    # Gaussian smoothing
    gaussian = cv2.GaussianBlur(grayscale, (3, 3), 0)
    cv2.imwrite('output/gaussian_img.jpg', gaussian)
    
    # Canny for edge detection
    edgesdetected = cv2.Canny(gaussian,50, 150,apertureSize = 3)
    cv2.imwrite('output/edgesdetected_img.jpg', edgesdetected)

    # Hough Transform
    minLineLen = 0
    maxLineGap = 0
    lines = cv2.HoughLinesP(edgesdetected, 1, np.pi/180, 100, minLineLen,maxLineGap)
    if detectOnlyPole:
        for array in lines:
            x1,y1,x2,y2 = array[0]
            if x1 < 600:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0), 4)
        cv2.imwrite('output/output_pole.jpg',image)
    else:
        for array in lines:
            x1,y1,x2,y2 = array[0]
            if x1 > 600:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0), 2)
            else:
                cv2.line(image,(x1,y1),(x2,y2),(255,0,0), 4)
        cv2.imwrite('output/output_all.jpg',image)
        
#Create object file
def create_object_file(df, pointcloudobj, intensity):
    object_df = pd.DataFrame()
    object_df[['x', 'y', 'z']] = df.apply(pd.Series)
    object_df['intensity'] = intensity
    object_df['v'] = 'v'
    object_df = object_df[['v', 'x', 'y', 'z', 'intensity']]
    object_df.to_csv(pointcloudobj, sep=' ', index=False, header=False)

#coordinate transformation for each point cloud
def convert_ecef_to_enu(row, R, x0, y0, z0):
    coordinates_diff = np.array([row['x'] - x0, row['y'] - y0, row['z'] - z0]).reshape(3, 1)
    res = R.dot(coordinates_diff).ravel().tolist()
    return res
    
# Rotation matrix for each point cloud to transform it to the position with respect to the camera's perspective, instead of with respect to the earth's origin
def create_rotation_matrix(row):
    latitude_radian = row['latitude']
    longitude_radian = row['longitude']
    latitude_sin = math.sin(latitude_radian)
    latitude_cos = math.cos(latitude_radian)
    longitude_sin = math.sin(longitude_radian)
    longitude_cos = math.cos(longitude_radian)
    RMat = np.array([
        [-longitude_sin, longitude_cos, 0],
        [-longitude_cos * latitude_sin, -latitude_sin * longitude_sin, latitude_cos],
        [latitude_cos * longitude_cos, latitude_cos * longitude_sin, latitude_sin]
    ])
    return RMat
 
#transform LLA to ECEF coordinate
def transform_lla_to_ecef(df):
    altitude = df['altitude'].values
    latitude = df['latitude'].values
    longitude = df['longitude'].values
    
    a = 6378137
    b = 6356752.3
    f = (a - b) / a
    e = math.sqrt(f * (2 - f))

    # compute N, distance from surface to z axis
    N = a / np.sqrt(1 - e ** 2 * np.sin(latitude) ** 2)

    # perform transformation to x, y, z axis according to the formula
    x = (altitude + N) * np.cos(latitude) * np.cos(longitude)
    y = (altitude + N) * np.cos(latitude) * np.sin(longitude)
    z = (altitude + (1 - e ** 2) * N) * np.sin(latitude)
    return x, y, z
    
def transform(point_clouds, cam_df):
    # Sliced point cloud wih intensity greater than intensity_thresold (40)
    point_clouds = point_clouds.loc[point_clouds['intensity'] >= intensity_thresold]
    point_clouds = point_clouds.reset_index(drop=True)

    # Transform LLA to ECEF coordinate
    x0, y0, z0 = transform_lla_to_ecef(cam_df)
    x, y, z = transform_lla_to_ecef(point_clouds)

    ecef = pd.DataFrame(np.column_stack((x, y, z)), columns=['x', 'y', 'z'])
    ecef[['latitude', 'longitude']] = point_clouds[['latitude', 'longitude']]

    # rotate coordinates wrt camera configuration
    R = create_rotation_matrix(cam_df)
    ecef['enu'] = ecef.apply(convert_ecef_to_enu, axis=1, args=(R, x0, y0, z0))

    # create 3d point cloud obj file
    create_object_file(ecef['enu'], 'output/pointcloud.obj', point_clouds['intensity'])
    
#convert degree to radian
def degree_to_radian(degree):
    radian = degree * math.pi / 180
    return radian
    
#read data and obtain point cloud and camera center.
def process_data_from_point_cloud():
    point_cloud_file = os.path.join('final_project_data', 'final_project_point_cloud.fuse')
    point_clouds = pd.read_csv(point_cloud_file, sep=',', header=None, names=['combined'])

    # split the combined column into separate columns
    point_clouds = point_clouds['combined'].str.split(' ', expand=True).astype(np.float64)
    point_clouds.columns = ['latitude', 'longitude', 'altitude', 'intensity']

    point_cloud_file = os.path.join('final_project_data', 'image', 'camera.config')
    with open(point_cloud_file, 'r') as f:
        f.readline()
        camera_conf_file = f.readline()

    # Reading camera configuration
    cam_center = [float(camera) for camera in camera_conf_file.split(', ')]
    cam_df = pd.DataFrame([cam_center[:3]], columns=['latitude', 'longitude', 'altitude'])

    cam_df['latitude'] = cam_df['latitude'].apply(degree_to_radian)
    cam_df['longitude'] = cam_df['longitude'].apply(degree_to_radian)
    point_clouds['latitude'] = point_clouds['latitude'].apply(degree_to_radian)
    point_clouds['longitude'] = point_clouds['longitude'].apply(degree_to_radian)

    transform(point_clouds, cam_df)
    
def main():
    print("Processing data from final_project_point_cloud.fuse")
    process_data_from_point_cloud()

    print("Generating binary image from point cloud")
    # Process data from pointcloud.obj file
    point_clouds = pd.read_csv('output/pointcloud.obj', sep=',', header=None, names=['combined'])

    point_clouds = point_clouds['combined'].str.split(' ', expand=True)
    point_clouds.columns = ['v','latitude', 'longitude', 'altitude', 'intensity']

    # Genearting image from point cloud
    fig = plt.figure(figsize=(30, 30))
    plt.plot(point_clouds['latitude'].tolist(), point_clouds['longitude'].tolist(), '.', color='k')
    plt.axis('off')
    fig.savefig("output/image.jpg", bbox_inches='tight')

    print("Detecting pole marking")
    detect_obj(cv2.imread('output/image.jpg'),True)
    print("Output image for pole detection saved: output/output_pole_detection.jpg")

    print("Detecting all line marking")
    detect_obj(cv2.imread('output/image.jpg'),False)
    print("Output image for all line detection saved: output/output_all_lines_detection.jpg")

main()