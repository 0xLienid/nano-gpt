import torch
from flask import Flask, request, jsonify
from bigram import BigramLanguageModel

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Here are all the unique characters in the text
chars = sorted(list(set(text)))

# Create a mapping from characters to integers
itos = { i:ch for i,ch in enumerate(chars) }
decode = lambda l: "".join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Load the model
model = BigramLanguageModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()  # set to eval mode

def convert_state_dict_to_json(state_dict):
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu().numpy().tolist()
    return state_dict

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    context = torch.zeros((1, 1), dtype=torch.long)

    with torch.no_grad():
        output = model.generate(context, max_new_tokens=500)[0].tolist()

    output = decode(output)  # Decode the output
    return jsonify(output)  # return the output as JSON

@app.route('/weights', methods=['GET'])
def weights():
    model_weights = model.state_dict()  # get the state dict
    jsonifiable_weights = convert_state_dict_to_json(model_weights)
    return jsonify(jsonifiable_weights)  # return weights as JSON

if __name__ == '__main__':
    app.run(debug=True, port=5000)
