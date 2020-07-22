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

try:
    diffusion, diffusion_n2i, diffusion_i2n, networks, node2tp, tp2node, node2ct = read_server_data(DATA_DIR)
    node_info, term2pid = read_node_info(DATA_DIR, tp2node)

except Exception as e:
    exit('load database error '+str(e))


app = flask.Flask(__name__)
app.config["DEBUG"] = True


defautstyle = [
            {
                "properties": {
                    "NETWORK_BACKGROUND_PAINT": "#FFFFFF",
                    "NETWORK_HEIGHT": "489.0",
                    "NETWORK_SIZE": "550.0",
                    "NETWORK_WIDTH": "795.0"
                },
                "properties_of": "network"
            },
            {
                "dependencies": {
                    "nodeCustomGraphicsSizeSync": "true",
                    "nodeSizeLocked": "false"
                },
                "mappings": {
                    "NODE_BORDER_PAINT": {
                        "definition": "COL=type,T=string,K=0=gene,V=0=#467CA8",
                        "type": "DISCRETE"
                    },
                    "NODE_BORDER_WIDTH": {
                        "definition": "COL=type,T=string,K=0=disease,V=0=3.0,K=1=gene,V=1=4.0",
                        "type": "DISCRETE"
                    },
                    "NODE_FILL_COLOR": {
                        "definition": "COL=type,T=string,K=0=disease,V=0=#FFFFFF,K=1=gene,V=1=#FFFFFF,K=2=function,V=2=#33CCDD,K=3=drug,V=3=#DAE10E",
                        "type": "DISCRETE"
                    },
                    "NODE_HEIGHT": {
                        "definition": "COL=type,T=string,K=0=disease,V=0=50.0,K=1=gene,V=1=25.0,K=2=function,V=2=20.0,K=3=drug,V=3=20.0",
                        "type": "DISCRETE"
                    },
                    "NODE_LABEL": {
                        "definition": "COL=name,T=string",
                        "type": "PASSTHROUGH"
                    },
                    "NODE_LABEL_COLOR": {
                        "definition": "COL=type,T=string,K=0=function,V=0=#111111",
                        "type": "DISCRETE"
                    },
                    "NODE_LABEL_POSITION": {
                        "definition": "COL=type,T=string,K=0=drug,V=0=S,,NW,,c,,0.00,,0.00",
                        "type": "DISCRETE"
                    },
                    "NODE_LABEL_WIDTH": {
                        "definition": "COL=type,T=string,K=0=disease,V=0=60.0",
                        "type": "DISCRETE"
                    },
                    "NODE_SHAPE": {
                        "definition": "COL=type,T=string,K=0=disease,V=0=HEXAGON,K=1=gene,V=1=ROUND_RECTANGLE,K=2=function,V=2=ROUND_RECTANGLE,K=3=drug,V=3=DIAMOND",
                        "type": "DISCRETE"
                    },
                    "NODE_WIDTH": {
                        "definition": "COL=type,T=string,K=0=disease,V=0=80.0,K=1=gene,V=1=60.0,K=2=function,V=2=60.0,K=3=drug,V=3=20.0",
                        "type": "DISCRETE"
                    }
                },
                "properties": {
                    "COMPOUND_NODE_PADDING": "10.0",
                    "COMPOUND_NODE_SHAPE": "ROUND_RECTANGLE",
                    "NODE_BORDER_PAINT": "#666666",
                    "NODE_BORDER_STROKE": "SOLID",
                    "NODE_BORDER_TRANSPARENCY": "255",
                    "NODE_BORDER_WIDTH": "0.0",
                    "NODE_CUSTOMGRAPHICS_1": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_2": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_3": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_4": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_5": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_6": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_7": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_8": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_9": "org.cytoscape.ding.customgraphics.NullCustomGraphics,0,[ Remove Graphics ],",
                    "NODE_CUSTOMGRAPHICS_POSITION_1": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_POSITION_2": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_POSITION_3": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_POSITION_4": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_POSITION_5": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_POSITION_6": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_POSITION_7": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_POSITION_8": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_POSITION_9": "C,C,c,0.00,0.00",
                    "NODE_CUSTOMGRAPHICS_SIZE_1": "50.0",
                    "NODE_CUSTOMGRAPHICS_SIZE_2": "50.0",
                    "NODE_CUSTOMGRAPHICS_SIZE_3": "50.0",
                    "NODE_CUSTOMGRAPHICS_SIZE_4": "50.0",
                    "NODE_CUSTOMGRAPHICS_SIZE_5": "50.0",
                    "NODE_CUSTOMGRAPHICS_SIZE_6": "50.0",
                    "NODE_CUSTOMGRAPHICS_SIZE_7": "50.0",
                    "NODE_CUSTOMGRAPHICS_SIZE_8": "50.0",
                    "NODE_CUSTOMGRAPHICS_SIZE_9": "50.0",
                    "NODE_CUSTOMPAINT_1": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_1, name=Node Custom Paint 1)",
                    "NODE_CUSTOMPAINT_2": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_2, name=Node Custom Paint 2)",
                    "NODE_CUSTOMPAINT_3": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_3, name=Node Custom Paint 3)",
                    "NODE_CUSTOMPAINT_4": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_4, name=Node Custom Paint 4)",
                    "NODE_CUSTOMPAINT_5": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_5, name=Node Custom Paint 5)",
                    "NODE_CUSTOMPAINT_6": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_6, name=Node Custom Paint 6)",
                    "NODE_CUSTOMPAINT_7": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_7, name=Node Custom Paint 7)",
                    "NODE_CUSTOMPAINT_8": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_8, name=Node Custom Paint 8)",
                    "NODE_CUSTOMPAINT_9": "DefaultVisualizableVisualProperty(id=NODE_CUSTOMPAINT_9, name=Node Custom Paint 9)",
                    "NODE_DEPTH": "0.0",
                    "NODE_FILL_COLOR": "#E5E5E5",
                    "NODE_HEIGHT": "40.0",
                    "NODE_LABEL_COLOR": "#262626",
                    "NODE_LABEL_FONT_FACE": "SansSerif,plain,12",
                    "NODE_LABEL_FONT_SIZE": "10",
                    "NODE_LABEL_POSITION": "C,C,c,0.00,0.00",
                    "NODE_LABEL_TRANSPARENCY": "255",
                    "NODE_LABEL_WIDTH": "200.0",
                    "NODE_NESTED_NETWORK_IMAGE_VISIBLE": "true",
                    "NODE_PAINT": "#1E90FF",
                    "NODE_SELECTED": "false",
                    "NODE_SELECTED_PAINT": "#FFFF00",
                    "NODE_SHAPE": "ROUND_RECTANGLE",
                    "NODE_SIZE": "35.0",
                    "NODE_TRANSPARENCY": "255",
                    "NODE_VISIBLE": "true",
                    "NODE_WIDTH": "40.0",
                    "NODE_X_LOCATION": "0.0",
                    "NODE_Y_LOCATION": "0.0",
                    "NODE_Z_LOCATION": "0.0"
                },
                "properties_of": "nodes:default"
            },
            {
                "dependencies": {
                    "arrowColorMatchesEdge": "false"
                },
                "properties": {
                    "EDGE_CURVED": "true",
                    "EDGE_LABEL_COLOR": "#000000",
                    "EDGE_LABEL_FONT_FACE": "Dialog,plain,10",
                    "EDGE_LABEL_FONT_SIZE": "10",
                    "EDGE_LABEL_TRANSPARENCY": "255",
                    "EDGE_LABEL_WIDTH": "200.0",
                    "EDGE_LINE_TYPE": "SOLID",
                    "EDGE_PAINT": "#323232",
                    "EDGE_SELECTED": "false",
                    "EDGE_SELECTED_PAINT": "#FF0000",
                    "EDGE_SOURCE_ARROW_SELECTED_PAINT": "#FFFF00",
                    "EDGE_SOURCE_ARROW_SHAPE": "NONE",
                    "EDGE_SOURCE_ARROW_SIZE": "6.0",
                    "EDGE_SOURCE_ARROW_UNSELECTED_PAINT": "#000000",
                    "EDGE_STROKE_SELECTED_PAINT": "#FF0000",
                    "EDGE_STROKE_UNSELECTED_PAINT": "#848484",
                    "EDGE_TARGET_ARROW_SELECTED_PAINT": "#FFFF00",
                    "EDGE_TARGET_ARROW_SHAPE": "NONE",
                    "EDGE_TARGET_ARROW_SIZE": "6.0",
                    "EDGE_TARGET_ARROW_UNSELECTED_PAINT": "#000000",
                    "EDGE_TRANSPARENCY": "255",
                    "EDGE_UNSELECTED_PAINT": "#404040",
                    "EDGE_VISIBLE": "true",
                    "EDGE_WIDTH": "2.0"
                },
                "properties_of": "edges:default"
            }
        ]

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
    cvt_nice_cx(nicecx)
    nicecx.opaqueAspects['cyVisualProperties'] = defautstyle
    rawcx = nicecx.to_cx()
    return jsonify(rawcx)


def cvt_nice_cx(nicecx):
    for key, value in nicecx.nodes.items():
        rep = value['n']
        value['r'] = rep
        node_id2, node_name, node_info2, title, description, url = query_node(rep, node_info, term2pid, node2tp, DATA_DIR)
        value['n'] = node_name
        #value['n'] = rep[3:].replace('_', ' ')
        node_attr_list = nicecx.nodeAttributes[key]
        node_attr_list.append({'po':key,"n": "title", 'v': title})
        node_attr_list.append({'po': key, 'n': 'description', 'v': description})
        node_attr_list.append({'po': key, 'n': 'url', 'v': url})




@app.route("/nodes", methods=['GET'])
def query_nodes():
    node_str = request.args.get('query')
    nodes = node_str.split(',')
    result = []
    for node in nodes:
        #node_id, node_info_tmp, title, description, url = query_node(node, node_info, term2pid, node2tp, DATA_DIR)
        node_id2, node_name, node_info2, title, description, url = query_node(node, node_info, term2pid, node2tp, DATA_DIR)
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
