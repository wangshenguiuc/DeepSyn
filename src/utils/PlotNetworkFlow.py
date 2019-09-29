from ortools.graph import pywrapgraph
from graphviz import Digraph

def write_network_cytoscape(output_file,node_set,edge_list,node_weight,word_type):
	fnode = open(output_file+'.node','w')
	fedge = open(output_file+'.edge','w')
	fnode.write('source\ttype\tsign\n')
	fedge.write('source\ttarget\tweight\tsource\n')
	for n in node_set:
		tp = word_type[n].split('_')[1]
		if tp=='gene':
			name = n.upper()
		else:
			name = n
		fnode.write(name+'\t'+tp+'\t'+str(node_weight[n]<0.05)+'\n')
	for e in edge_list:
		e1,e2,w,source = e
		source = source.replace('infer','deep')
		sl = source.split('_')
		for s in sl:
			fedge.write(e1+'\t'+e2+'\t'+str(w)+'\t'+s+'\n')
	fnode.close()
	fedge.close()



def plot_network_flow(output_file,s,node_set,edge_list,node_weight, word_ct, word_type, tgt_set=[]):
	dot = Digraph(comment='Network flow')
	dot.format = 'eps'

	for node in node_set:
		nwt = node_weight[node]
		if float(nwt) < 0.01:
			nwt = '%.2e' % (float(nwt) )
		else:
			nwt = '%.2f' % (float(nwt) )
		if node in word_ct['pubmed']:
			wct = word_ct['pubmed'][node]
		else:
			wct = -1
		if node==s:
			dot.attr('node',color='blue')
			dot.node(node + ' '+ str(wct)+ ' ' + str(nwt))
		elif node in word_type and node in tgt_set: #word_type[node].split('_')[1] == 'gene':
			dot.attr('node',color='red')
			dot.node(node + ' '+ str(wct)+ ' ' + str(nwt))
		elif node in word_type and word_type[node].split('_')[1] == 'function':
			dot.attr('node',color='green')
			dot.node(node + ' '+ str(wct)+ ' ' + str(nwt))
		elif node in word_type and word_type[node].split('_')[1] == 'disease':
			dot.attr('node',color='yellow')
			dot.node(node + ' '+ str(wct)+ ' ' + str(nwt))
		elif node in tgt_set:
			dot.attr('node',color='blue')
			dot.node(node + ' '+ str(wct)+ ' ' + str(nwt))
		else:
			dot.attr('node',color='black')
			dot.node(node + ' '+ str(wct)+ ' ' + str(nwt))
	for e in edge_list:
		e1,e2,w,source = e
		source = source.replace('_','&')
		source = source.replace('infer','deep')
		wct1 = word_ct['pubmed'].get(e1,-1)
		wct2 = word_ct['pubmed'].get(e2,-1)
		nwt1 = node_weight[e1]
		nwt2 = node_weight[e2]
		if float(nwt1) < 0.01:
			nwt1 = '%.2e' % (float(nwt1) )
		else:
			nwt1 = '%.2f' % (float(nwt1) )
		if float(nwt2) < 0.01:
			nwt2 = '%.2e' % (float(nwt2) )
		else:
			nwt2 = '%.2f' % (float(nwt2) )
		if e1 not in word_type:
			t1 = 'entity'
		else:
			t1 = word_type[e1].split('_')[1]
		if e2 not in word_type:
			t2 = 'entity'
		else:
			t2 = word_type[e2].split('_')[1]

		#type = '_'.join(type_list)
		if float(w) < 0.01:
			wt = '%.2e' % (float(w) )
		else:
			wt = '%.2f' % (float(w) )
		dot.edge(e1+ ' '+ str(wct1)+ ' ' + str(nwt1),e2+ ' '+ str(wct2)+ ' ' + str(nwt2),label=source+':'+t1+'->'+t2+'('+wt+')')

	dot.render(output_file)
