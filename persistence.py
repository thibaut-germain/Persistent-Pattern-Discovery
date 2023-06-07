import numpy as np 
import plotly.graph_objects as go

class BasicPersistence(object): 

    def __init__(self): 
        pass

    def make_set(self,x:int,w:float)->None: 
        """Add node x at its birth

        Args:
            x (int): node id
            w (float): rank set as the node birth
        """
        self.parent_node_[x] = x
        self.rank_[x] = w

    def find(self,x:int)->int: 
        """look for the parent node of node x.

        Args:
            x (int): child node id

        Returns:
            int: parent node id
        """
        if self.parent_node_[x]!=x: 
            self.parent_node_[x] = self.find(self.parent_node_[x])
        return self.parent_node_[x]
        

    def fit(self,filtration:np.ndarray)->None:
        """ Compute graph homology persistence of degree zero.

        Args:
            filtration (np.ndarray): birth increasing order filtration, shape: (n_connexions,3) -> n_connexions * {left node,rigth node, birth date}.
        """
        self.parent_node_ = {}
        self.rank_ = {}
        self.persistence_ =[]
        self.mst_ =[]

        for x,y,w in filtration: 

            try:
                u = self.find(x)
            except: 
                self.make_set(x,w)
                u = x
            try: 
                v = self.find(y)
            except: 
                self.make_set(y,w)
                v = y

            if u != v:
                birth = max(self.rank_[u],self.rank_[v])
                if self.rank_[u]<self.rank_[v]:
                    cluster_id = self.parent_node_[v] 
                    self.parent_node_[v] = u
                else:
                    cluster_id = self.parent_node_[u]
                    self.parent_node_[u] = v
                self.persistence_.append([birth,w,cluster_id])
                self.mst_.append([x,y,w])

        for u in self.parent_node_.keys(): 
            self.find(u)

        self.persistence_ = np.array(self.persistence_)
        self.mst_ = np.array(self.mst_)
        return self

    def get_persistence(self,with_infinite_point = True): 
        if with_infinite_point: 
            infinite_points = np.unique(list(self.parent_node_.values()))
            lst = []
            for point in infinite_points: 
                lst.append([self.rank_[point],np.inf,point])
            return np.vstack((self.persistence_,lst))
        else: 
            return self.persistence_


    def persistence_diagram(self,min_persistence = 0.,persistence_threshold = None,birth_threshold = None)->go.Figure: 
        """ display persistence diagram

        Args:
            min_persistence (float, optional): only display connected object with a persistence higher than min_persistence. Defaults to 0..
            persistence_threshold (float, optional): blur persitence diagram area whose birth is less than persistence_threshold . Defaults to None.
            birth_threshold (float, optional): blur persitence diagram area whose birth is higher than birth_threshold. Defaults to None.

        Returns:
            go.Figure: persistence diagram
        """
        fig = go.Figure()
        # add diagonals
        max_size = np.max(self.persistence_,axis=0)
        max_h = 1.05
        max_v = 1.2
        fig.update_layout(xaxis_range = [0,max_h*max_size[0]],yaxis_range = [0,max_v*max_size[1]])
        fig.add_trace(go.Scatter(x=[0,max_h*max_size[0],max_h*max_size[0],0], y = [0,max_h*max_size[0],0,0],fill = 'toself',marker=dict(size=1),hovertemplate='<extra></extra>'))
        fig.add_trace(go.Scatter(x=[0,max_h*max_size[0]], y = [0,max_h*max_size[0]],mode='lines', marker=dict(color ='black'),hovertemplate='<extra></extra>'))

        #add infinity bar
        infty_val = (1+max_v)/2
        fig.add_trace(go.Scatter(x=[0,max_h*max_size[0]], y = [infty_val*max_size[1],infty_val*max_size[1]],mode='lines', marker=dict(color ='black'),hovertemplate='<extra></extra>'))

        # add non infinity points
        idxs, = np.where(self.persistence_[:,1]-self.persistence_[:,0] > min_persistence)
        z = self.persistence_[idxs,1]-self.persistence_[idxs,0]
        hovertemplate = 'persistence: %{text:.2f} <br>birth: %{x:.2f} <br>death: %{y:.2f} <extra></extra>'
        fig.add_trace(go.Scatter(x= self.persistence_[idxs,0],y= self.persistence_[idxs,1],mode='markers',marker=dict(color = 'red'),text = z,hovertemplate=hovertemplate,showlegend=False))
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)

        #add infinity points
        infty_points = np.unique(list(self.parent_node_.values()))
        infty_points = [self.rank_[infty_point] for infty_point in infty_points]
        hovertemplate = 'persistence: infinity  <br>birth: %{x:.2f} <br>death: infinity <extra></extra>'
        fig.add_trace(go.Scatter(x= infty_points, y = np.zeros_like(infty_points)+infty_val*max_size[1], mode='markers', marker=dict(color = 'red'),hovertemplate=hovertemplate))
        
        #add persistence threshold
        if persistence_threshold is not None: 
            fig.add_trace(go.Scatter(x=[0,max_h*max_size[0],max_h*max_size[0],0], y = [0+persistence_threshold,max_h*max_size[0]+persistence_threshold,0,0],fill = 'toself',marker=dict(size=1,color='grey'),hovertemplate='<extra></extra>'))

        #add birth threshold
        if birth_threshold is not None: 
            fig.add_trace(go.Scatter(x = [birth_threshold,birth_threshold,max_h*max_size[0],max_h*max_size[0]], y = [0,max_v*max_size[1],max_v*max_size[1],0], fill = 'toself', marker=dict(size=1,color="grey"),hovertemplate='<extra></extra>'))

        # add annotations
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title = 'Birth',yaxis_title ='Death', title = 'Persistence Diagram')

        return fig  

#########################################################################################################################################
#########################################################################################################################################

class ThresholdPersistenceMST(object): 

    def __init__(self,persistence_threshold = np.inf,birth_threshold = np.inf, birth_individual_threshold = None)->None: 
        """Initialization of Threshold Persistence MST

        Args:
            persistence_threshold (float, optional): persistence threshold. Connected object of higher persistence the persistence threshold are of interest. Defaults to np.inf.
            birth_threshold (float, optional): birth threshold. Connected objects of whose birth are less than birth_threshold are of interest. Defaults to np.inf.
        """
        self.persistence_threshold = persistence_threshold
        self.birth_threshold = birth_threshold
        self.birth_individual_threshold = birth_individual_threshold

    def make_set(self,x:int,w:float)->None: 
        """Add node x at its birth

        Args:
            x (int): node id
            w (float): rank set as the node birth
        """
        self.parent_node_[x] = x
        self.rank_[x] = w

    def find(self,x:int)->int: 
        """look for the parent node of node x.

        Args:
            x (int): child node id

        Returns:
            int: parent node id
        """
        if self.parent_node_[x]!=x: 
            self.parent_node_[x] = self.find(self.parent_node_[x])
        return self.parent_node_[x]

    def fit(self,mst:np.ndarray)->None: 
        """ Compute graph homology persistence of degree zero.

        Args:
            filtration (np.ndarray): birth increasing order filtration, shape: (n_connexions,3) -> n_connexions * {left node,rigth node, birth date}.
        """
        self.parent_node_ = {}
        self.rank_ = {}
        self.mst_ = mst

        for x,y,w in mst: 

            try:
                u = self.find(x)
            except: 
                self.make_set(x,w)
                u = x
            try: 
                v = self.find(y)
            except: 
                self.make_set(y,w)
                v = y

            if u != v:
                birth = max(self.rank_[u],self.rank_[v])
                if w-birth < self.persistence_threshold: 
                    if self.rank_[u]<self.rank_[v]: 
                        self.parent_node_[v] = u
                    else:
                        self.parent_node_[u] = v
        
        for u in self.parent_node_.keys(): 
            self.find(u)

        self._build_connected_components()

        return self 

    def _build_connected_components(self)->None: 
        """Compute minimum connected objects
        """
        # get connected component before birth cut
        arr = np.array([[key,val] for key,val in self.parent_node_.items()]).astype(int)
        arr = arr[arr[:,1].argsort()]
        seeds = np.unique(arr[:,1])
        ccps = np.split(arr[:,0], np.unique(arr[:,1],return_index=True)[1][1:])

        #remove nodes based on birth threshold
        rank = np.array([(key,value) for key,value in self.rank_.items()])
        rank = rank[rank[:,0].argsort()]
        self.connected_components_ = dict()
        valid_ccps = rank[seeds,1]<self.birth_threshold
        self.seeds_ = seeds[valid_ccps]
        
        for seed,ccp in zip(seeds[valid_ccps],np.array(ccps,dtype=object)[valid_ccps]): 
            try:
                idx = np.where(rank[ccp.astype(int),1]<self.birth_individual_threshold[seed])
            except:
                idx = np.where(rank[ccp.astype(int),1]<self.birth_threshold)
            self.connected_components_[seed] = ccp[idx]