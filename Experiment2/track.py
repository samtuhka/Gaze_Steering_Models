import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rotate(px, py, angle, ox, oy):

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def road(axes, centrelane = True, turns = 20):
    xr = []
    yr = []

    xr_i = []
    yr_i = []

    xr_o = []
    yr_o = []

    s = 0
    last_x, last_y = -50,0
    for i in range(turns):
        s = np.deg2rad(-30)
        e =  np.deg2rad(120) 
        rang = np.linspace(s, s + e, 1000)
        x = np.sin(rang)*50
        y = np.cos(rang)*50


            

        xi = np.sin(rang)*48.25
        yi = np.cos(rang)*48.25
        
        xo = np.sin(rang)*51.75
        yo = np.cos(rang)*51.75


        if i % 2 == 0:
            xi, yi = rotate(xi, yi, np.pi, 0, 0)
            xi, yi  = xi[::-1], yi[::-1]
            xo, yo = rotate(xo, yo, np.pi, 0, 0)
            xo, yo  = xo[::-1], yo[::-1]
            x, y = rotate(x, y, np.pi, 0, 0)
            x, y = x[::-1], y[::-1]
        #if i % 2 == 0:
        #    rang = np.linspace(np.pi - s, np.pi - s + np.deg2rad(12), 100)
        #    x = np.cos(rang)
        #    y = np.sin(rang)
        xo += last_x - x[0]
        yo += last_y - y[0]

        xi += last_x - x[0]
        yi += last_y - y[0]

        x += last_x - x[0]
        y += last_y - y[0]



        #if i % 2 == 0:
        #    x, y, = rotate(np.pi, x, y)
        #    x = x[::-1]
        #    y = y[::-1]

        #plt.plot(x[-1], y[-1], 'og')
        #plt.plot(x[0], y[0], 'or')

        last_x = x[-1]
        last_y = y[-1]
        xr += x.tolist()
        yr += y.tolist()

        if turns == 20:
            road = np.load("road_full.npy")
            waypoints = road[::130]
            for ax in axes:
                for t in waypoints:
                    #print(t)
                    circle = plt.Circle((t[0], t[1]),0.7, color = 'blue', alpha = 0.8, fill = False, zorder=10)
                    ax.add_patch(circle)

        if i % 2 == 0:
            xr_i += xi.tolist()
            yr_i += yi.tolist()

            xr_o += xo.tolist()
            yr_o += yo.tolist()
        else:
            xr_i += xo.tolist()
            yr_i += yo.tolist()

            xr_o += xi.tolist()
            yr_o += yi.tolist()

    xr = np.array(xr)#*50 - 800
    yr = np.array(yr)#*50
    xr_i = np.array(xr_i)#*50 - 800
    yr_i = np.array(yr_i)#*50
    xr_o = np.array(xr_o)#*50 - 800
    yr_o = np.array(yr_o)#*50

    for ax in axes:
        if centrelane: 
            ax.plot(xr - 800, yr, '-r')
            ax.plot(xr_i - 800, yr_i, '--', color = 'gray')
            ax.plot(xr_o - 800, yr_o, '--', color = 'gray')
        else:
            ax.plot(xr_i - 800, yr_i, 'k--')
            ax.plot(xr_o - 800, yr_o, 'k--')


#fig0 = plt.figure(figsize=(3, 5))
mean_path = np.load("./non_model_data/path.npy")
road([plt.gca()])

plt.gca().set_aspect(1)
arrow = patches.FancyArrowPatch((mean_path[0,0] + 10, mean_path[0,1]), (mean_path[260,0] + 10, mean_path[260,1]),
                         connectionstyle="arc3,rad=.14",  color="k", arrowstyle = "Simple, tail_width=0.5, head_width=4, head_length=8")
plt.gca().add_patch(arrow)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.xlabel("x-coord (m)")
plt.ylabel("y-coord (m)")
plt.ylim(-160, 0)
plt.xlim(-850, -690)
plt.tight_layout()
plt.savefig("track.pdf")
plt.savefig("track.png")
#plt.plot(obstacles[:,0], obstacles[:,2], 'ro')
plt.show()
