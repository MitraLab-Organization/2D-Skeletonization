import json
import numpy as np

def cc_to_json(cc):
    """
    Converts connected components data to a GeoJSON format.
    """
    data_out = {
        'type': 'FeatureCollection',
        'features': []
    }

    for i in range(cc['NumObjects']):
        feature = {
            'type': 'Feature',
            'ID': i,
            'properties': {
                'stroke_width': 1
            },
            'geometry': {},
            'length': 0
        }

        line_path = cc['arcProperties'][i]
        
        if len(line_path) == 1:
            if line_path[0]['length'] > 0:
                feature['geometry']['type'] = 'LineString'
                coords = np.zeros_like(line_path[0]['Pos'])
                coords[:, 0] = line_path[0]['Pos'][:, 0]
                coords[:, 1] = line_path[0]['Pos'][:, 1] * -1
                feature['geometry']['coordinates'] = coords.tolist()
                feature['length'] = line_path[0]['length']
        else:
            feature['geometry']['type'] = 'MultiLineString'
            coordinates = []
            length_n = 0
            for idx in range(len(line_path)):
                coords = np.zeros_like(line_path[idx]['Pos'])
                coords[:, 0] = line_path[idx]['Pos'][:, 0]
                coords[:, 1] = line_path[idx]['Pos'][:, 1] * -1
                coordinates.append(coords.tolist())
                length_n += line_path[idx]['length']
            
            feature['geometry']['coordinates'] = coordinates
            feature['length'] = length_n

        data_out['features'].append(feature)

    return data_out