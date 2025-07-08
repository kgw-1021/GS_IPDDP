import torch
import numpy as np
import open3d as o3d
import scipy
import time

class GS_IPDDP():
    def __init__(self, gsplat, robot_config, env_config, spline_planner, device):
        # gsplat: GSplat object

        self.gsplat = gsplat
        self.device = device

        # Robot configuration
        self.radius = robot_config['radius']
        self.vmax = robot_config['vmax']
        self.amax = robot_config['amax']
        self.collision_set = GSplatCollisionSet(self.gsplat, self.vmax, self.amax, self.radius, self.device)

        # Environment configuration (specifically voxel)
        self.lower_bound = env_config['lower_bound']
        self.upper_bound = env_config['upper_bound']
        self.resolution = env_config['resolution']

        tnow = time.time()
        torch.cuda.synchronize()
        self.gsplat_voxel = GSplatVoxel(self.gsplat, lower_bound=self.lower_bound, upper_bound=self.upper_bound, resolution=self.resolution, radius=self.radius, device=device)
        torch.cuda.synchronize()
        print('Time to create GSplatVoxel:', time.time() - tnow)

        # Spline planner
        self.spline_planner = spline_planner

        # Save the mesh
        # gsplat_voxel.create_mesh(save_path=save_path)
        # gsplat.save_mesh(scene_name + '_gsplat.obj')

        # Record times
        self.times_cbf = []
        self.times_qp = []
        self.times_prune = []

    def generate_path(self, x0, xf):
        # Part 1: Computes the path seed using A*
        tnow = time.time()
        torch.cuda.synchronize()

        path = self.gsplat_voxel.create_path(x0, xf)

        torch.cuda.synchronize()
        time_astar = time.time() - tnow

        times_collision_set = 0
        times_polytope = 0

        polytopes = []      # List of polytopes (A, b)
        segments = torch.tensor(np.stack([path[:-1], path[1:]], axis=1), device=self.device)

        for it, segment in enumerate(segments):

            # Test if the current segment is in the most recent polytope
            if it > 0:
                is_in_polytope = compute_segment_in_polytope(polytope[0], polytope[1], segment)
            else:
                #If we haven't created a polytope yet, so we set it to False.
                is_in_polytope = False

            # If this is the first line segment, we always create a polytope. Or subsequently, we only instantiate a polytope if the line segment
            if (it == 0) or (it == len(segments) - 1) or (not is_in_polytope):

                # Part 2: Computes the collision set
                tnow = time.time()
                torch.cuda.synchronize()
                
                output = self.collision_set.compute_set_one_step(segment)

                torch.cuda.synchronize()
                times_collision_set += time.time() - tnow

                # Part 3: Computes the polytope
                tnow = time.time()
                torch.cuda.synchronize()

                polytope = self.get_polytope_from_outputs(output)

                torch.cuda.synchronize()
                times_polytope += time.time() - tnow

                polytopes.append(polytope)

        # Step 4: Perform Bezier spline optimization
        tnow = time.time()
        torch.cuda.synchronize()

        traj, feasible = self.spline_planner.optimize_b_spline(polytopes, segments[0][0], segments[-1][-1])
        if not feasible:
            traj = torch.stack([x0, xf], dim=0)

            self.save_polytope(polytopes, 'infeasible.obj')

            print(compute_segment_in_polytope(polytope[0], polytope[1], segments[-1]))
            raise

        torch.cuda.synchronize()
        times_opt = time.time() - tnow
  
        # Save outgoing information
        traj_data = {
            'path': path.tolist(),
            'polytopes': [torch.cat([polytope[0], polytope[1].unsqueeze(-1)], dim=-1).tolist() for polytope in polytopes],
            'num_polytopes': len(polytopes),
            'traj': traj.tolist(),
            'times_astar': time_astar,
            'times_collision_set': times_collision_set,
            'times_polytope': times_polytope,
            'times_opt': times_opt,
            'feasible': feasible
        }

        # self.save_polytope(polytopes, 'feasible.obj')
        
        return traj_data
    

