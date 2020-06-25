import flask
from flask import request, jsonify
from utils import *
import ndex2
import configparser

# Read configuration
config = configparser.ConfigParser()
config.read('/Users/jingchen/git/DeepSyn/api.ini')
# config.read('api.ini')
s = config.sections()
localConfig = config['DeepSyn']
DATA_DIR = localConfig['data_dir']
REST_PORT = localConfig['port']

diffusion, diffusion_n2i, diffusion_i2n, networks, node2tp, tp2node, node2ct = read_server_data(DATA_DIR)
node_info, term2pid = read_node_info(DATA_DIR)

app = flask.Flask(__name__)
app.config["DEBUG"] = True


def main():
    print("Starting REST server.")
    app.run(port=REST_PORT)


@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "online"})


@app.route("/query", methods=['POST'])
def query_genes():
    query = request.get_json()
    ans_paths, ans_nodes = run_query(query, networks, diffusion, diffusion_n2i, diffusion_i2n, node2tp, MAX_DEPTH=4)
    print(ans_paths)
    nx_obj = create_networkx_obj(ans_paths, ans_nodes, node2tp)
    nicecx = ndex2.create_nice_cx_from_networkx(nx_obj)
    rawcx = nicecx.to_cx()
    return jsonify(rawcx)


@app.route("/nodes", methods=['GET'])
def query_nodes():
    node_str = request.args.get('query')
    nodes = node_str.split(',')
    result = []
    for node in nodes:
        node_info_tmp, title, description, url = query_node(node, node_info, term2pid, node2tp, DATA_DIR)
        n_holder = {'id': node, 'title': title, 'description': description, 'url': url}
        result.append(n_holder)
    return jsonify(result)


@app.route("/edges", methods=['GET'])
def query_edges():
    edge_str = request.args.get('query')
    edges = edge_str.split(',')
    result = []
    for path in edges:
        path_info = query_edge(path, DATA_DIR)
        n_holder = {'id': path, 'info': path_info}
        result.append(n_holder)
    return jsonify(result)


if __name__ == "__main__":
    main()
