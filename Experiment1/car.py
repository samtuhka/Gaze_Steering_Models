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

    def pursuitsteer(self, point, dt, weight = 1):

        point = toEgo(point, self.pos(), self.rot + np.pi)

        rotdt_ideal =  purepursuit_world(point, self.speed)

        self.goal += rotdt_ideal*weight

    def move(self, dt, smooth = 1, clip = 35):

        self.rotdt = (self.goal)*smooth + (1 - smooth)*self.rotdt
        self.rotdt = np.clip(self.rotdt, np.deg2rad(-clip), np.deg2rad(clip))
        self.rot += self.rotdt*dt
        self.x += self.speed*dt*(np.sin(self.rot))
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

    return proj

def purepursuit_world(steerpoint, v):
    x,y = steerpoint
    yaw = -(2*v*x)/(y**2 + x**2)
    return yaw

def toProj(p):
    h = 1.5
    hrad = np.arctan(p[0]/p[1])
    vrad = np.arctan(h/p[1])
    return [hrad, vrad]
