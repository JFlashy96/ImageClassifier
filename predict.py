import argparse
import helper

parser = argparse.ArgumentParser(
	description = ("Parser for predict.py")
	)

parser.add_argument("--input", default="./flower_data/test/1/image_06752.jpg", nargs="?", action="store", type=str)
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./flower_data")
parser.add_argument("--output_number", default=5, action="store", type=int)
parser.add_argument("--arch", action="store",                    
                             choices=["vgg11", "vgg11_bn", "vgg13", "vgg13_bn",
                             "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"],
                             default="vgg19")
parser.add_argument("--checkpoint", default="./checkpoint.pth", action="store", type=str)
parser.add_argument("--gpu", default="cuda", action="store", dest="gpu")

args = parser.parse_args()

predict_image = args.input
data_dir = args.data_dir
arch = args.arch
num_outputs = args.output_number
checkpoint = args.checkpoint
device = args.gpu
model = helper.load_checkpoint(checkpoint, arch, device)

def main():

    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
        
    probabilities = helper.predict(predict_image, model, num_outputs, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])

    i=0

    while i < num_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Finished Predicting!")

if __name__ == "__main__":
	main()