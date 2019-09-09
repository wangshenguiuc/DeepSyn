import matplotlib.pyplot as plt
import matplotlib
import os
class 2DPlot_tsne():
	def __init__(self, Xvec,Yvec,fig_dir = '',filename = '',Xlabel='',Ylabel='',title='',figsize=15,cmap='Spectral',marker='o',
	c='g',node_label=[],add_diagnoal = True,node_label_cutoff = 0):
		tmp_node_label = []
		if type(Xvec) is dict:
			Xvec_l = []
			Yvec_l = []
			for d in Xvec:
				if d in Yvec:
					Xvec_l.append(Xvec[d])
					Yvec_l.append(Yvec[d])
					if type(node_label) is dict and len(node_label) > 0:
						tmp_node_label.append(node_label[d])
					else:
						tmp_node_label.append(d)
			Xvec = Xvec_l
			Yvec = Yvec_l
		plt.figure(figsize=(figsize,figsize))
		plt.title(title)
		#plt.subplots_adjust(bottom = 0.1)

		plt.scatter(
			Xvec, Yvec, marker=marker, c=c,
			cmap=plt.get_cmap(cmap))
		if type(node_label) is not list:
			node_label = tmp_node_label
		if len(node_label) == len(Xvec):
			for label, x, y in zip(node_label,Xvec,Yvec):
				if abs(x) > node_label_cutoff or abs(y) > node_label_cutoff:
					label_z = label
				else:
					label_z = ''
				plt.annotate(
					label_z,
					xy=(x, y), xytext=(0,5),
					textcoords='offset points', ha='right', va='bottom')
		if add_diagnoal:
			diagnoal_max = max(max(Xvec),max(Yvec))
			diagnoal_min = min(min(Xvec),min(Yvec))
			plt.plot( [diagnoal_min,diagnoal_max],[diagnoal_min,diagnoal_max] ,ls='--')
		plt.xlabel(Xlabel)
		plt.ylabel(Ylabel)

		plt.legend(loc=2)
		#plt.show()
		fig_name = fig_dir + filename
		if len(fig_name) > 0:
			if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)
			plt.savefig(fig_name)
			plt.close('all')
		else:
			plt.show()
