def main(p):
    "print pickle"
    print(p)
    # p = '/home/rhagoletis/catkin/src/World/bags/2017_01_28/2017-01-28__03:30:35_mdom8_n_pf_gain:8_speed:1.0_bout:3_DC:0.0_traj.pickle'
    # from helping.helper import depickler
    from World.helping.helper import depickler
    a = depickler(p)


 # ...
if __name__ == '__main__':
 import plac; plac.call(main)