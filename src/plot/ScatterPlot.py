import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
class ScatterPlot():
	def __init__(self, Xvec,Yvec,fig_dir = '',filename = '',Xlabel='',Ylabel='',title='',figsize=15,cmap='Spectral',marker='o',
	c='g',node_label=[],add_diagnoal = True,node_label_cutoff = 0,legend_loc = 2,label_font=20):
		tmp_node_label = []
		if type(Xvec) is dict:
			Xvec_l = []
			Yvec_l = []
			dlist = []
			for d in Xvec:
				if d in Yvec:
					Xvec_l.append(Xvec[d])
					Yvec_l.append(Yvec[d])
					if type(node_label) is dict and len(node_label) > 0:
						tmp_node_label.append(node_label[d])
					else:
						tmp_node_label.append(d)
					dlist.append(d)
			Xvec = Xvec_l
			Yvec = Yvec_l
		dcolor = []
		Xvec = np.array(Xvec)
		Yvec = np.array(Yvec)
		#print c
		if type(c) is dict:
			for d in dlist:
				if d in c:
					dcolor.append(c[d])
				else:
					dcolor.append('missing')
		else:
			for i in range(len(Xvec)):
				dcolor.append(c)
		color_set, colors = np.unique(dcolor, return_inverse=True)
		tis2col = dict(zip(dcolor, colors))
		ncolors = len(color_set)
		plt.figure(figsize=(figsize,figsize))
		plt.title(title,fontsize = label_font)
		#print tis2col
		#plt.subplots_adjust(bottom = 0.1)
		dcolor = np.array(dcolor)
		#color_list = ['b','c','y','m','r']
		color_list = plt.cm.rainbow(np.linspace(0, 1, ncolors))
		sct_l = []
		label_l = []
		for c in tis2col:
			#print c,tis2col[c]
			ind = np.where(dcolor==c)[0]
			col_vec=np.repeat(tis2col[c], len(ind))
			#print ind,col_vec
			sct = plt.scatter(
				Xvec[ind], Yvec[ind], marker=marker, c=color_list[tis2col[c]],
				cmap=plt.get_cmap(cmap))
			sct_l.append(sct)
			label_l.append(c)
		plt.legend(sct_l,label_l
           ,
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=16)
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
		plt.xlabel(Xlabel,fontsize = label_font)
		plt.ylabel(Ylabel,fontsize = label_font)

		plt.legend(loc=legend_loc)
		#plt.show()
		fig_name = fig_dir + filename
		if len(fig_name) > 0:
			if not os.path.exists(fig_dir):
				os.makedirs(fig_dir)
			plt.savefig(fig_name)
			plt.close('all')
		else:
			plt.show()
