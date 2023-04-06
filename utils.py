import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import kubric as kb
import os 
import traitlets as tl
## curves 

def visualize_curve(f, start, stop):
    ts = np.linspace(start, stop, num=100)
    plt.plot(ts, [f(t) for t in ts])

def get_1D_curve_from_func(f, input_start, input_end, target_start, target_end):
    a = (target_end - target_start) / (f(input_end) - f(input_start))
    b = target_end - a*f(input_end)
    return lambda t: a * f((input_end - input_start) * t + input_start) + b

def straight_1D_curve(target_start, target_end): 
    return lambda t: (target_end - target_start) * t + target_start
    
def previsualize_scene(camera_curve, objects_curve):
    ts = np.linspace(0, 1, 100)
    plt.plot(ts, [objects_curve(t) for t in ts], label='Objects')
    plt.plot(ts, [camera_curve(t) for t in ts], label='Camera')
    plt.legend()



def random_speed_jumps(nb_jumps): 
    ys = [0]
    for _ in range(nb_jumps-1):
        ys.append(np.random.uniform(low=ys[-1] + 0.05, high=ys[-1] + 1))

    return np.array(ys) / ys[-1]


def varying_dynamics(nb_jumps, nb_points):
    xs = np.linspace(0,1,nb_jumps)
    ys = random_speed_jumps(nb_jumps)
    return np.interp(np.linspace(0,1,nb_points), xs, ys)


def curve_and_positions_3D(
                        boundaries,
                        shape_func_along_axis_and_input_boundaries,
                        axis):
    
    curves = []
    for ax in range(3):
        boundaries_axis = boundaries[ax]
        if ax == axis:
            curves.append(get_1D_curve_from_func(
                                    shape_func_along_axis_and_input_boundaries[0], 
                                    *shape_func_along_axis_and_input_boundaries[1],
                                    *boundaries_axis))
        else: 
            curves.append(straight_1D_curve(*boundaries_axis))


    return curve_3D_from_1D_curves(*curves)

                        
def curve_3D_from_1D_curves(curve_x, curve_y, curve_z):
    return lambda t: np.array([curve_x(t), curve_y(t), curve_z(t)])

## positionnings 
def regular_positionning_along_curve(
                    curve, 
                    num_positions, 
                    rng=None, 
                    noise_scales=(0.1, 0, 0.1)):
    
    positions = np.array([curve(t) for t in np.linspace(
                                        start=0, 
                                        stop=1, 
                                        num=num_positions)])
    if rng is not None:
        positions = add_noise(rng, positions, noise_scales)
        
    return positions

def position_along_curve_with_varying_speed(
                    curve, 
                    num_positions, 
                    num_speed_jumps,
                    rng=None, 
                    noise_scales=(0.1, 0, 0.1)):
    
    ts = varying_dynamics(num_speed_jumps, num_positions)
    

    positions = np.array([curve(t) for t in ts])
    if rng is not None:
        positions = add_noise(rng, positions, noise_scales)
        
    return positions, ts

def positionning_along_curve(curve, dynamics, num_positions):
    return np.array([curve(dynamics(t)) for t in np.linspace(0,1, num_positions)])

def add_noise(rng:np.random.RandomState, positions, noise_scales):

    if noise_scales[0] != 0:
        noise_x = rng.normal(loc=0, scale=noise_scales[0], size=len(positions))
    else: noise_x = np.zeros(shape=(len(positions),))

    if noise_scales[1] != 0: 
        noise_y = rng.normal(loc=0, scale=noise_scales[1], size=len(positions))
    else: noise_y = np.zeros(shape=(len(positions),))

    if noise_scales[2] != 0:
        noise_z = rng.normal(loc=0, scale=noise_scales[2], size=len(positions))
    else: noise_z = np.zeros(shape=(len(positions),))

    return positions + np.array([noise_x, noise_y, noise_z]).T

## scales 

def random_scales(num_objects, rng:np.random.RandomState):
    random_scales = rng.normal(loc=0.2, scale=0.1, size=num_objects)
    return np.clip(random_scales, 0.05, 0.95)

## preview scene

def euler_to_cartesian(euler_orientation): 
    alpha, beta, gamma = euler_orientation
    x = np.cos(np.pi / 2 + gamma)
    y = np.sin(np.pi / 2 + gamma)
    return np.array([x,y,0])

def preview_scene_along_axis(
                camera_curve,
                camera_positions,
                camera_orientations,
                objects_curve,
                objects_positions,
                objects_scales,
                axis):
    

    timesteps_for_curves = np.linspace(0,1,1000)
    camera_curve_sampled_along_axis = np.array([(camera_curve(t)[0], camera_curve(t)[axis]) for t in timesteps_for_curves])
    objects_curve_sampled_along_axis = np.array([(camera_curve(t)[0], objects_curve(t)[axis]) for t in timesteps_for_curves])
    objects_positions_along_axis = np.array([(pos[0], pos[axis]) for pos in objects_positions])
    camera_positions_along_axis = np.array([(pos[0], pos[axis]) for pos in camera_positions])
    plt.plot(camera_curve_sampled_along_axis[:,0], camera_curve_sampled_along_axis[:,1], label='Camera curve')
    plt.scatter(camera_positions_along_axis[:,0], camera_positions_along_axis[:,1], label='Camera positions')
    camera_orientations_cartesian = np.array([euler_to_cartesian(euler_coord) for euler_coord in camera_orientations])
    
    for (x,y,dx,dy) in zip(
                        camera_positions_along_axis[:,0], 
                        camera_positions_along_axis[:,1], 
                        camera_orientations_cartesian[:,0], 
                        camera_orientations_cartesian[:,1]):
        u = np.array((dx, dy))
        dx, dy = u / np.linalg.norm(u)
        plt.gca().add_patch(patches.Arrow(x,y,dx,dy,width=0.05))

    plt.plot(objects_curve_sampled_along_axis[:,0], objects_curve_sampled_along_axis[:,1], label='Objects curve')
    plt.scatter(objects_positions_along_axis[:,0], objects_positions_along_axis[:,1], label='Objects positions')
    for x, y, scale in zip(objects_positions_along_axis[:,0], objects_positions_along_axis[:,1], objects_scales): 
        plt.gca().add_patch(patches.Circle((x,y), scale / 2, fill=False, color='C1'))
    plt.gca().set_aspect('equal')
    # plt.gca().invert_yaxis()
    # plt.gca().xaxis.tick_top()
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.autoscale(True)
    plt.tight_layout()
    # plt.legend()




def add_objects(scene: kb.Scene, 
                positions, 
                scales):
    
    for position, scale in zip(positions, scales):
        obj = kb.Sphere(position)
        obj.material = kb.PrincipledBSDFMaterial(color=kb.Color(1.0, 0.0, 0.0))
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])

        scene += obj
        
    return scene


def add_positions_for_camera(scene:kb.Scene,
                            positions):

    for frame_nb, position in enumerate(positions):
        scene.camera.position = position
        scene.camera.keyframe_insert("position", frame_nb)
        scene.camera.keyframe_insert("quaternion", frame_nb)

    return scene


def visibility_masks_from_segmentations(segmentations, path=None):
    
    num_frames = segmentations.shape[0]

    object_ids_in_video = []
    visibility_masks = dict()

    object_ids_in_prev_frame = []
    N_plus = np.zeros((num_frames,))
    N_minus = np.zeros((num_frames,))

    for frame_nb, frame in enumerate(segmentations):
        object_ids_in_frame = np.unique(frame)

        for object_id in object_ids_in_frame:
            if object_id not in object_ids_in_prev_frame:
                N_plus[frame_nb] += 1
                if object_id not in object_ids_in_video: 
                    object_ids_in_video.append(object_id)
                    visibility_masks[object_id] = np.zeros((num_frames,))
                
        for object_id in object_ids_in_prev_frame:
            if object_id not in object_ids_in_frame:
                N_minus[frame_nb] +=1

        for object_id in object_ids_in_video:
            if object_id in object_ids_in_frame:
                visibility_masks[object_id][frame_nb] = 1
        object_ids_in_prev_frame = object_ids_in_frame

    # N_plus[0] -= 2 
    # visibility_masks.pop(1)
    # visibility_masks.pop(2)

    if path is not None: 
        plt.close()
        for k,v in visibility_masks.items():
            plt.step(np.arange(num_frames), v, label=k)
        plt.savefig(os.path.join(path, 'visibility_masks'))
        plt.close()

        plt.step(np.arange(num_frames), N_plus, label='N+')
        plt.step(np.arange(num_frames), N_minus, label='N-')
        plt.legend()
        plt.savefig(os.path.join(path,'N_plus_N_minus'))
        plt.close()
        
    return N_plus, N_minus, visibility_masks

if __name__ == '__main__':

    test = varying_dynamics(3, np.random.default_rng())
    x = np.linspace(0,1,100)
    plt.plot(x, [test(t) for t in x])
    plt.savefig("test")
