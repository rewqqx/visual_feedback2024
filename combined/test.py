import pybullet as p
import numpy as np
import time
import matplotlib.pyplot as plt
from camera import Camera
import cv2

IMG_SIDE = 300
IMG_HALF = IMG_SIDE / 2
MARKER_LENGTH = 0.1
MARKER_CORNERS_WORLD = np.array(
    [
        [-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0.0, 1],
        [MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0.0, 1],
        [MARKER_LENGTH / 2, -MARKER_LENGTH / 2.0, 0.0, 1],
        [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0.0, 1]
    ]
)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)


def computeInterMatrix(Z, sd0):
    L = np.zeros((8, 3))
    for idx in range(4):
        x = sd0[2 * idx, 0]
        y = sd0[2 * idx + 1, 0]
        L[2 * idx] = np.array([-1 / Z, 0, y])
        L[2 * idx + 1] = np.array([0, -1 / Z, -x])
    return L


def updateCamPos(camera, angle):
    linkState = p.getLinkState(boxId, linkIndex=6)
    # pos
    xyz = linkState[0]
    # orientation
    quat = p.getQuaternionFromEuler([0, 0, angle])
    rotMat = p.getMatrixFromQuaternion(quat)
    rotMat = np.reshape(np.array(rotMat), (3, 3))
    camera.set_new_position(xyz, rotMat)


def move_eef_to_pos(
        desired_pos,
        mode="iterative",
        maxNumIterations=10,
        xy_plot=False,
        sleep=False,
        verbose_pos=False,
):
    assert mode in ["iterative", "direct"]
    xs, ys = [], []

    def append_xys():
        currentPose = p.getLinkState(boxId, eefLinkIdx)
        x, y, _ = currentPose[0]
        xs.append(x)
        ys.append(y)

    def print_pos():
        currentPose = p.getLinkState(boxId, eefLinkIdx)
        x, y, z = currentPose[0]
        print(f"x y z = {x} {y} {z}")

    append_xys() if xy_plot else None

    if mode == "iterative":
        for _ in range(100):
            jointPoses = p.calculateInverseKinematics(boxId, eefLinkIdx, desired_pos, maxNumIterations=maxNumIterations)
            p.setJointMotorControlArray(
                bodyIndex=boxId,
                jointIndices=jointIndices,
                targetPositions=jointPoses,
                controlMode=p.POSITION_CONTROL
            )
            p.stepSimulation()
            updateCamPos(camera, 0)
            camera.get_frame()
            time.sleep(0.15) if sleep else None
            append_xys() if xy_plot else None
            print_pos() if verbose_pos else None
    else:
        jointPoses = p.calculateInverseKinematics(boxId, eefLinkIdx, desired_pos, maxNumIterations=maxNumIterations)
        p.setJointMotorControlArray(
            bodyIndex=boxId,
            jointIndices=jointIndices,
            targetPositions=jointPoses,
            controlMode=p.POSITION_CONTROL
        )
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.15) if sleep else None
            append_xys() if xy_plot else None
            print_pos() if verbose_pos else None

    if xy_plot:
        plt.figure()
        plt.title(f"InverseKinematics mode: {mode}")
        plt.plot(xs, ys, '--')
        plt.scatter(xs[0], ys[0], marker='o', color="green", label="starting position")
        plt.scatter(desired_pos[0], desired_pos[1], marker='o', color="red", label="desired position")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.legend()
        plt.show()


camera = Camera(imgSize=[IMG_SIDE, IMG_SIDE])

dt = 1 / 240  # pybullet simulation step
q0 = 0.5  # starting position (radian)
qd = 0.5

xd = 0.5
yd = 0.5

L = 0.5
Z0 = 0.3
pos = q0
maxTime = 10
logTime = np.arange(0.0, maxTime, dt)
sz = logTime.size
logPos = np.zeros(sz)
logPos[0] = q0
logVel = np.zeros(sz)

jointIndices = [1, 3, 5]
eefLinkIdx = 6

# or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI,
                          options="--background_color_red=1 --background_color_blue=1 --background_color_green=1")
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=-90,
    cameraPitch=-89.999,
    cameraTargetPosition=[0.5, 0.5, 0.6]
)
p.setGravity(0, 0, -10)
boxId = p.loadURDF("./simple.urdf.xml", useFixedBase=True)

# add aruco cube and aruco texture
markerPos = (0.5, 0.5, 0.0)
c = p.loadURDF('aruco.urdf', markerPos, useFixedBase=True)
x = p.loadTexture('aruco_cube.png')
p.changeVisualShape(c, -1, textureUniqueId=x)

numJoints = p.getNumJoints(boxId)
for idx in range(numJoints):
    print(f"{idx} {p.getJointInfo(boxId, idx)[1]} {p.getJointInfo(boxId, idx)[12]}")

# go to the desired position
currentPose = p.getLinkState(boxId, eefLinkIdx)
currentPosition = currentPose[0]
desired_pos = np.array([markerPos[0], markerPos[1], currentPosition[2]])
move_eef_to_pos(
    desired_pos,
    mode="iterative",
    maxNumIterations=100,
    xy_plot=True,
    sleep=False,
    verbose_pos=True
)

updateCamPos(camera, 0)
img = camera.get_frame()
corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
sd0 = np.reshape(np.array(corners[0][0]), (8, 1))
sd0 = np.array([(s - IMG_HALF) / IMG_HALF for s in sd0])
sd = np.reshape(np.array(corners[0][0]), (8, 1)).astype(int)

# go to the starting position
random_xy_offset = np.random.rand(2) / 50
starting_pos = desired_pos + np.append(random_xy_offset, 0)
move_eef_to_pos(
    starting_pos,
    mode="iterative",
    maxNumIterations=100,
    xy_plot=False,
    sleep=True,
    verbose_pos=True
)

idx = 1
camCount = 0
w = np.zeros((3, 1))
for t in logTime[1:]:
    p.stepSimulation()

    camCount += 1
    if (camCount == 5):
        camCount = 0
        cam_angle = t * np.pi / 10  # Adjust the angle as desired
        updateCamPos(camera, cam_angle)
        img = camera.get_frame()
        corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
        s = corners[0][0, 0]
        s0 = np.reshape(np.array(corners[0][0]), (8, 1))
        s0 = np.array([(ss - IMG_HALF) / IMG_HALF for ss in s0])
        L0 = computeInterMatrix(Z0, s0)
        L0T = np.linalg.inv(L0.T @ L0) @ L0.T
        e = s0 - sd0
        coef = 1 / 2
        w = -coef * L0T @ e

    jStates = p.getJointStates(boxId, jointIndices=jointIndices)
    jPos = [state[0] for state in jStates]
    jVel = [state[1] for state in jStates]
    (linJac, angJac) = p.calculateJacobian(
        bodyUniqueId=boxId,
        linkIndex=6,
        localPosition=[0, 0, 0],
        objPositions=jPos,
        objVelocities=[0, 0, 0],
        objAccelerations=[0, 0, 0]
    )

    J = np.block([
        [np.array(linJac)[:2, :2], np.zeros((2, 1))],
        [np.array(angJac)[2, :]]
    ])
    dq = (np.linalg.inv(J) @ w).flatten()[[1, 0, 2]]
    dq[2] = -dq[2]
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=dq,
                                controlMode=p.VELOCITY_CONTROL)
    # time.sleep(0.01)

p.disconnect()