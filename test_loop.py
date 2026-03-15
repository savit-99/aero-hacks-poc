import numpy as np
import math

V_BASE = 5.0
DT = 0.1

mines = np.array([[250.0, 250.0]])
waypoints = [np.array([250.0, 300.0])] # Target is directly north of the mine
current_pos = np.array([250.0, 245.0]) # Starting directly south, on the boundary!
wp_idx = 0
circled_mines = {0} # Already circled!

steps = 0
while wp_idx < len(waypoints):
    steps += 1
    if steps > 1000:
        print('Infinite loop! current_pos:', current_pos)
        break
        
    target_pos = waypoints[wp_idx]
    dist = np.linalg.norm(target_pos - current_pos)
    
    if dist < (V_BASE * DT):
        current_pos = target_pos
        wp_idx += 1
        continue
        
    direction = (target_pos - current_pos) / dist
    next_pos = current_pos + direction * (V_BASE * DT)
    
    dists_to_mines = np.linalg.norm(mines - next_pos, axis=1)
    min_dist = np.min(dists_to_mines)
    closest_mine_idx = np.argmin(dists_to_mines)
    
    if min_dist < 5.0:
        mine_pos = mines[closest_mine_idx]
        if closest_mine_idx not in circled_mines:
            pass # We don't enter here
        else:
            escape_vec = next_pos - mine_pos
            escape_dist = max(np.linalg.norm(escape_vec), 0.001)
            projected_pos = mine_pos + (escape_vec / escape_dist) * 5.0
            
            if np.linalg.norm(projected_pos - current_pos) < 0.05:
                # Nudge it sideways to force a slide!
                tangent = np.array([-escape_vec[1], escape_vec[0]])
                tangent = tangent / max(np.linalg.norm(tangent), 0.001)
                projected_pos = current_pos + tangent * (V_BASE * DT)
                escape_vec2 = projected_pos - mine_pos
                projected_pos = mine_pos + (escape_vec2 / max(np.linalg.norm(escape_vec2), 0.001)) * 5.0
                
            next_pos = projected_pos
            
    current_pos = next_pos
    if steps < 5:
        print('Step:', steps, 'pos:', current_pos)
        
print("Finished in", steps, "steps. Final pos:", current_pos)
