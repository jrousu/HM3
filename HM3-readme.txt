Hierarchical Max-Margin Markov Algorithm implementation (c) 2004-2007 Juho Rousu, 
This package contains the MATLAB implementation of the 
Hierarchical Max-Margin Markov Algorithm, as described in the manuscript
accepted to Journal of Machine Learning Research.
'Rousu, Saunders, Szedmak and Shawe-Taylor: 
Kernel-Based Learning of Hierarchical Multilabel Classification Models'

The implementation is due to Juho Rousu (firstname dot lastname at
aalto.fi). The implementation is a research prototype and is provided AS
IS, no warranties or quarantees of any kind are given. 

Data format
===========

Kx_tr				m by m x-kernel matrix of the training points
Kx_ts				m by m' x-kernel matrix of the test points, m' is the number of test points

X_tr				m by n matrix of training point features, where n is the number of features associated with each training point
X_ts				m' by n matrix of test point features, where n is the number of features associated with each training point

Y_tr				m by k matrix containing the multilabels of the training points
Y_ts				m' by k	matrix containing the multilabels of the testing points 

E				|E| x 4 matrix containing the edge set of the hierarchy, (E(i,1),E(i,2))
				contains the i'th edge as microlabel indices, E(i,3) is the running index of
				edge, E(i,4) is the depth of the edge in the hierarchy
					
				Important note: due to the way the inference algorithm has been implemented,
				the global root of the hierarchy should only have a single child. Easiest
				way of ensuring this is to add a new dummy root node to the hierarchy and
				edge from that to the current root. (This is a bit ugly, but was simpler
				to implement this way)

params				Input parameter structure
	.x_datasource		Where x-data is to be found: 0 = kernel Kx_tr in memory, 
				1 = matrix X_tr, 2 = load from disk
	.max_ip_iter		maximum number of interior point iterations for LIPSOL 
        .profiling		1 = produce profiling information during learning, 0 = do not
    	.profile_tm_interval	Time interval between profile computations in seconds (default 600 sec)
        .profile_log		File to output the profile info
	.verbosity		Verbosity level
	.tolerance		The tolerance for numerical inaccuracy
	.epsilon		Stopping criterion, typically maximum allowed relative duality gap
	.chunksize		How many examples to process at a time
	.enforce_tree		Only constrain margins of labelings that represent trees (relaxed dual, so
				approximates the original
	.tree_inference		Whether to predict labelings consistent with the tree or arbitrary multilabels
	.slack			Slack parameter C
	.maxiter		Maximum number of conditional gradient iterations
	.ws_workmem_limit	How much working mememory to use
	.hloss			Use hierarhical loss (mistake in child penalized only if parent was correct)
	.scaling		Scaling of the loss 0 = no scaling, 1 = scaling by number of siblings, 2 =
				scaling by size of the subtree below

Contents
========

Main programs
-------------

learn_M3_maxtree_cgd		The main algorithm for learning HM^3, in chunked form where
				a set of examples is simultaneously updated
learn_M3_maxtree_cgd_decomp	The main algorithm for learning HM^3, in decomposed form where
				a single example is updated at a time [lower memory footprint]
learn_M3_LP_cgd_decomp		The same but using LP instead of dynamic programming for 
				conditional gradient (the algorithm in the ICML'05 paper)				
								
Update directions and inference
-------------------------------

max_tree_direction		Computes the update direction for the 
				marginalized dual (the conditional gradient) 
				using dynamic programming inference on the tree
max_tree_direction_x		Same as above but for a single example (used by the maxtree_cgd_decomp version)

max_LP_direction_x		Same as above but via solving an LP (used by the LP_cgd_decomp version)
				
max_gradient_labeling		The dynamic programming search for max gradient
				labelings, used both by max_tree_direction
				and test_M3 (with different gradients)
max_gradient_labeling_x		Same as above but for a single example (used by the maxtree_cgd_decomp version)

compute_w_phi_e			Computes the gradient vector required for
				obtaining max-likelihood labeling from
				max_gradient_labeling_ls

				

Matrix vector products
----------------------
				
compute_Kmu			Efficiently computes the product K \mu between
				the marginalized joint kernel and the marginal
				dual vector.				
compute_Kmu_x			Same as above but for a single example i: [K \mu]_i (used by the _decomp version)				

compute_Kmu_1			Efficiently computes the product K v, vhere v is a
				vertex of the feasible set given by the conditional gradient.
				Used for speeding up saddle point computations
				[in the _decomp version the same code is inlined, hence no separate routine
				'compute_Kmu_1x' or the like]				

Profiling
---------

compute_duality_gap_vector	Computes the duality gap and its approximate
				division to the training example

test_M3				Tests the accuracy of the model with the current
				test set

compute_M3_error		Tests the accuracy of the model during
				optimization	
			

Learning problem setup
----------------------	

get_x_kernel			A stub for obtaining the x-kernel
load_x_kernel			Load the kernel from the disk (not tested)


compute_loss_vector		For obtaining the vector of losses in the
compute_loss_scaling		marginalized dual objective

form_consistency_constraints.m	Sets up the constraint matrices (used by learn_M3_LP_cgd_decomp)
form_box_constraints_x.m				

compute_tree_size		Computes the size of a subtree

find_edge_neighbours		Finds the parent and the children of a node	

precomputeIndicatorMatrices	Computes indicator matrices representing the 
				4x4 tiles that make up the joint kernel matrix.
				Used for speed-up purposes

compute_domain_sizes		Computes domain sizes of node-labelings
compute_edge_domain_sizes	Precomputes domain size of edge-labelings
				Note: the code is in some places optimized
				not to refer to these tables but assumes
				binary node-domains and 4-value edge-domains
				This of course should be corrected.

M3_set_globals			Sets most of the global variables used by the 
				algorithms. Does not set the kernels, data and
				the edges: Kx_tr and Kx_ts (or X_tr and X_ts), Y_tr, Y_ts and E need to
				be set global before starting the algorithm

binary_var_count		computes the number of edge-dual variables
