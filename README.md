# fusion_perception

ROS package for fusion perception (camera & lidar & radar)

## 安装
 - 使用Anaconda设置环境依赖，并确保系统已经安装ROS
   ```
   # 创建名为yolact-env的虚拟环境
   conda create -n yolact-env python=3.6.9
   conda activate yolact-env
   
   # 安装PyTorch、torchvision以及其他功能包
   conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
   pip install cython
   pip install opencv-python pillow pycocotools matplotlib
   pip install -U scikit-learn
   
   # 使Python3与ROS兼容
   pip install catkin_tools
   pip install rospkg
   ```
 - 建立工作空间并拷贝这个库
   ```Shell
   mkdir -p ros_ws/src
   cd ros_ws/src
   git clone https://github.com/shangjie-li/fusion_perception.git --recursive
   git clone https://github.com/shangjie-li/points_process.git
   git clone https://github.com/shangjie-li/points_ground_filter.git
   git clone https://github.com/shangjie-li/perception_msgs.git
   cd ..
   catkin_make
   ```
 - 下载模型文件[yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)，并保存至目录`fusion_perception/modules/yolact/weights`

## 参数配置
 - 编写相机标定参数`fusion_perception/conf/calibration_image.yaml`
   ```
   %YAML:1.0
   ---
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [461, 0, 333, 0, 0, 463, 184, 0, 0, 0, 1, 0]
   ```
 - 编写激光雷达标定参数`fusion_perception/conf/calibration_lidar.yaml`
   ```
   %YAML:1.0
   ---
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [461, 0, 333, 0, 0, 463, 184, 0, 0, 0, 1, 0]
   SensorToCameraMat: !!opencv-matrix
      rows: 4
      cols: 4
      dt: d
      data: [0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
   RotationAngleX: -1
   RotationAngleY: -1
   RotationAngleZ: 0
   ```
 - 编写毫米波雷达标定参数`fusion_perception/conf/calibration_radar.yaml`
   ```
   %YAML:1.0
   ---
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [461, 0, 333, 0, 0, 463, 184, 0, 0, 0, 1, 0]
   SensorToCameraMat: !!opencv-matrix
      rows: 4
      cols: 4
      dt: d
      data: [0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
   RotationAngleX: -1
   RotationAngleY: -1
   RotationAngleZ: 0
   ```
 - 在上述三个参数文件中
   ```
   ProjectionMat:
     该3x4矩阵为通过相机内参标定得到的projection_matrix。
   SensorToCameraMat:
     该4x4矩阵为坐标系的齐次转换矩阵，左上角3x3为旋转矩阵，右上角3x1为平移矩阵。
     例如：
     Translation = [dx, dy, dz]T
                [0, -1,  0]
     Rotation = [0,  0, -1]
                [1,  0,  0]
     则：
     SensorToCameraMat = [0, -1,  0, dx]
                         [0,  0, -1, dy]
                         [1,  0,  0, dz]
                         [0,  0,  0,  1]
   RotationAngleX/Y/Z:
     该值是对SensorToCameraMat矩阵进行修正的旋转角度，初始应设置为0，之后根据投影效果进行细微调整，单位为度。
   ```
 - 修改目标检测及跟踪算法相关参数`fusion_perception/conf/param.yaml`
   ```
   print_time:                         True
   print_objects_info:                 False
   record_time:                        True
   record_objects_info:                True
  
   use_lidar:                          True
   use_radar:                          True
  
   sensor_height:                      2.0 # meter
   sensor_depression:                  0.0 # deg
   rotation_lidar_to_front:            0.0 # deg
   rotation_radar_to_front:            0.0 # deg
  
   sub_image_topic:                    /usb_cam/image_raw
   sub_lidar_topic:                    /lidar_points_no_ground
   sub_radar_topic:                    /radar
   pub_marker_topic:                   /objects
   pub_obstacle_topic:                 /obstacles
   frame_id:                           /pandar
  
   calibration_image_file:             calibration_image.yaml
   calibration_lidar_file:             calibration_lidar.yaml
   calibration_radar_file:             calibration_radar.yaml
  
   display_image_raw:                  False
   display_image_segmented:            False
  
   display_lidar_projected:            False
   display_radar_projected:            False
  
   display_2d_modeling:                False
   display_gate:                       False
  
   display_3d_modeling:                True
   display_frame:                      True
   display_obj_state:                  True
  
   processing_mode: DT # D - detection, DT - detection and tracking
  
   blind_update_limit:                 0
   frame_rate:                         10
   max_id:                             10000
   ```
    - `sensor_height`为感知平台的高度，单位米。
    - `sensor_depression`为感知平台的俯角，单位度。
    - `rotation_lidar_to_front`为激光雷达X轴旋转至感知平台正前方的角度，单位度。
    - `rotation_radar_to_front`为毫米波雷达X轴旋转至感知平台正前方的角度，单位度。
    - `sub_image_topic`指明订阅的图像话题。
    - `sub_lidar_topic`指明订阅的激光雷达话题。
    - `sub_radar_topic`指明订阅的毫米波雷达话题。
    - `pub_marker_topic`指明发布的话题，类型为`MarkerArray`，可以通过`rviz`查看。
    - `pub_obstacle_topic`指明发布的话题，类型为`ObstacleArray`。

## 运行
 - 启动点云预处理节点`points_process`和地面滤波节点`points_ground_filter`
   ```
   cd ros_ws
   source devel/setup.bash
   
   # Don't forget to adjust parameters in the launch file
   roslaunch points_process points_process.launch
   
   # Don't forget to adjust parameters in the launch file
   roslaunch points_ground_filter points_ground_filter.launch
   ```
 - 加载参数文件至ROS参数服务器
   ```
   cd fusion_perception/conf
   rosparam load param.yaml
   ```
 - 启动目标检测和跟踪节点`detection_and_tracking`
   ```
   cd fusion_perception/scripts
   python3 detection_and_tracking.py
   ```

## 附图
 - 下图为激光雷达和毫米波雷达标定文件中的坐标关系示意图
   ```
                               \     /    Rotation:
                                \ |z/     [0 -1  0]
                                 \|/      [0  0 -1]
                                  █————x  [1  0  0]
                               forward    
                                 cam_1    
  
                                █████
                   |x         ██  |x ██
   [1  0  0]       |         █    |    █                 [-1 0  0]
   [0  0 -1]  z————█ cam_4  █ y———.z    █  cam_2 █————z  [0  0 -1]
   [0  1  0]                 █         █         |       [0 -1  0]
                              ██     ██          |x      
                                █████
                                sensor

                             x————█       [0  1  0]
                                  |       [0  0 -1]
                                  |z      [-1 0  0]
                                 cam_3    
   ```


