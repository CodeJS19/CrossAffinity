from inference_util import parse_csv
import multiprocessing
import argparse
import time
start = time.time()

worker_lock = None
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Inference script")
    
    parser.add_argument("--filepath", type=str, help="Filepath to input csv file.")
    parser.add_argument("--output", type=str, default='Predicted_Affinity.csv', help="Output file name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--cross_affinity_device", type=str, default='cpu', help="Device to use CrossAffinity on. e.g. 'cpu', 'cuda:0'")
    parser.add_argument("--esm2_device", type=str, default='cpu', help="Device to use ESM2 on. e.g. 'cpu', 'cuda:0'")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for parallelisation.")
    parser.add_argument("--model_name", type=str, default='1280_128_1_2_2_32_0_0.001', help="Model name to use. Default uses pre-trained CrossAffinity model.")
    parser.add_argument("--model_path", type=str, default='./published_model_weights/', help="Path to model. Default uses path to pre-trained CrossAffinity model.")
    
    args = parser.parse_args()
    filepath, batch_size, cross_affinity_device, esm2_device, model_path, model_name, num_workers, output = args.filepath, args.batch_size, args.cross_affinity_device, args.esm2_device, args.model_path, args.model_name, args.num_workers, args.output
    
    parse_csv(filepath, batch_size, cross_affinity_device, esm2_device, model_path, model_name, num_workers, output)
    print('It took', time.time()-start, 'seconds.')
