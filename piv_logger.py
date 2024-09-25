import csv
import json
import os

class PIVLogger:
    def __init__(self, session_folder):
        self.csv_file_path = os.path.join(session_folder, 'piv_data.csv')
        self.json_file_path = os.path.join(session_folder, 'piv_data.json')
        self.create_csv()

    def create_csv(self):
        with open(self.csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "X", "Y", "U", "V", "Mean Magnitude", "Mean Vorticity", "Reynolds Stress"])

    def log_to_csv(self, timestamp, x, y, u, v, magnitude, vorticity, reynolds_stress):
        # Flatten arrays before storing
        x_flat = x.flatten()
        y_flat = y.flatten()
        u_flat = u.flatten()
        v_flat = v.flatten()
        
        with open(self.csv_file_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, x_flat, y_flat, u_flat, v_flat, np.mean(magnitude), np.mean(vorticity), reynolds_stress])

    def log_to_json(self, timestamp, x, y, u, v):
        # Log data into a JSON file
        data = {
            "timestamp": timestamp,
            "x": x.tolist(),
            "y": y.tolist(),
            "u": u.tolist(),
            "v": v.tolist()
        }
        with open(self.json_file_path, 'a') as jsonfile:
            json.dump(data, jsonfile)
            jsonfile.write("\n")
