import numpy as np


class Car:

    def __init__(self, x, y, rot = 0):
        self.x = x
        self.y = y
        self.rot = rot
        self.rotdt = 0
        self.speed = 0
        self.goal = 0

    def pos(self):
        return [self.x, self.y]

    def fixedyaw(self, yaw, dt):
        self.rotdt = yaw
        self.rot += self.rotdt*dt 
        self.x -= self.speed*dt*(np.sin(self.rot))
        self.y += self.speed*dt*(np.cos(self.rot))

    def pursuitsteer_alt(self, point, dt, weight = 1):


        rotdt_ideal =  purepursuit_screen(point, self.speed)

        #rotdt_sign = np.sign(rotdt_ideal - self.rotdt)
        #maxFrameRot = np.deg2rad()
        #rotdt_ideal = min(abs(rotdt_ideal - self.rotdt), maxFrameRot)*rotdt_sign
        self.goal += rotdt_ideal*weight #min(abs(rotdt_ideal - self.rotdt), maxFrameRot)*rotdt_sign*weight

    def pursuitsteer(self, point, dt, weight = 1):

        point = toEgo(point, self.pos(), -self.rot)

        rotdt_ideal =  purepursuit_world(point, self.speed)

        #rotdt_sign = np.sign(rotdt_ideal - self.rotdt)
        #maxFrameRot = np.deg2rad()
        #rotdt_ideal = min(abs(rotdt_ideal - self.rotdt), maxFrameRot)*rotdt_sign
        self.goal += rotdt_ideal*weight #min(abs(rotdt_ideal - self.rotdt), maxFrameRot)*rotdt_sign*weight

    def move(self, dt, smooth = 1, frameRot = 10000):
        rotdt_sign = np.sign(self.goal - self.rotdt)
        maxFrameRot = np.deg2rad(100000*dt)

        #self.rotdt = self.goal
        self.rotdt += min(abs(self.goal - self.rotdt), maxFrameRot)*rotdt_sign*smooth
        #self.rotdt = np.clip(self.rotdt, np.deg2rad(-35), np.deg2rad(35))
        self.rot += self.rotdt*dt
        self.x -= self.speed*dt*(np.sin(self.rot))
        self.y += self.speed*dt*(np.cos(self.rot))
        self.goal = 0


def rotate(px, py, angle, ox = 0, oy = 0):
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def salvucci_gray(near, far, dt):
    kf = 1 #0.5
    kn = 0
    k1 = 0
    yaw_change = kf * far + kn*near + k1*near*dt
    yaw_rate = yaw_change/dt
    return yaw_rate


def isBehind(proj, pos, rotation):
    pos = list(pos)
    proj = list(proj)
    pos = toEgo(proj, pos, rotation)
    hrad = np.arctan(pos[0]/pos[1])
    vrad = np.arctan(1.5/pos[1])
    if vrad > 0:
        return False
    return True

def toEgo(proj, pos, rotation):
    #plt.sca(axes[1]) 
    proj = [proj[0], proj[1]]
    pos = [pos[0], pos[1]]
    proj[0], proj[1] = rotate(proj[0], proj[1], rotation, pos[0], pos[1])
    proj[0] -= pos[0]
    proj[1] -= pos[1]


    #ego_p.set_data(proj[0], proj[1])
    #plt.xlim(-30,30)
    #plt.ylim(0,20)
    #plt.sca(axes[0]) 

    return proj

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def purepursuit_screen(steerpoint, v):
    H, V = steerpoint

    H, V = np.deg2rad(H), np.deg2rad(V)

    h = 1.2
    v = 10.5
    y = -np.sin(2*H)*np.tan(V)*v/h
    #y = np.rad2deg(y)
    return y

def purepursuit_world(steerpoint, v):
    x,y = steerpoint
    yaw = -(2*v*x)/(y**2 + x**2)
    return yaw

def toProj(p):
    h = 1.5
    hrad = np.arctan(p[0]/p[1])
    vrad = np.arctan(h/p[1])
    return [hrad, vrad]
