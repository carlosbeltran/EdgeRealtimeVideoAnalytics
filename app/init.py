# RedisEdge realtime video analytics initialization script
import argparse
import redis
from urllib.parse import urlparse

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='CPU or GPU', type=str, default='CPU')
    parser.add_argument('-i', '--camera_id', help='Input video stream key camera ID', type=str, default='0')
    parser.add_argument('-p', '--camera_prefix', help='Input video stream key prefix', type=str, default='camera')
    parser.add_argument('-u', '--url', help='RedisEdge URL', type=str, default='redis://127.0.0.1:6379')
    args = parser.parse_args()

    # Set up some vars
    input_stream_key = '{}:{}'.format(args.camera_prefix, args.camera_id)  # Input video stream key name
    initialized_key = '{}:initialized'.format(input_stream_key)

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    # Check if this Redis instance had already been initialized
    initialized = conn.exists(initialized_key)
    if initialized:
        print('Discovered evidence of a privious initialization - skipping.')
        exit(0)

    # Load the RedisAI model
    print('Loading model - ', end='')
    with open('models/tiny-yolo-voc.pb', 'rb') as f:
        model = f.read()
        res = conn.execute_command('AI.MODELSET', 'yolo:model', 'TF', args.device, 'INPUTS', 'input', 'OUTPUTS', 'output', model)
        print(res)

    # Load the PyTorch post processing boxes script
    print('Loading script - ', end='')
    with open('yolo_boxes.py', 'rb') as f:
        script = f.read()
        res = conn.execute_command('AI.SCRIPTSET', 'yolo:script', args.device, script)
        print(res)

    # Load the gear
    print('Loading gear - ', end='')
    with open('gear.py', 'rb') as f:
        gear = f.read()
        res = conn.execute_command('RG.PYEXECUTE', gear)
        print(res)

    # Lastly, set a key that indicates initialization has been performed
    print('Flag initialization as done - ', end='')
    print(conn.set(initialized_key, 'most certainly.'))
