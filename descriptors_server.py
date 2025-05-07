from flask import Flask, request, jsonify
import torch
import numpy as np
import open3d as o3d
from gedi import GeDi

app = Flask(__name__)

@app.route('/compute_gedi_descriptors', methods=['POST'])
def compute_descriptors():
    try:
        # Get point cloud data from request
        data = request.get_json()
        point_cloud = np.array(data['point_cloud'])
        radius = data['r_lrf']

        # Configuration for GeDi
        config = {
            'dim': 32,
            'samples_per_batch': 500,
            'samples_per_patch_lrf': 4000,
            'samples_per_patch_out': 512,
            'r_lrf': radius,
            'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'
        }
        gedi = GeDi(config=config)

        # Convert point cloud to tensor
        pcd_tensor = torch.tensor(point_cloud).float()

        with torch.no_grad():
            # Compute descriptors
            descriptors = gedi.compute(pts= pcd_tensor, pcd= pcd_tensor)

        # Return descriptors as JSON
        return jsonify({"descriptors": descriptors.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
