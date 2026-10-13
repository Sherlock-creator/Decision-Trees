from __future__ import annotations
import pandas as pd
import math
from abc import ABC, abstractmethod
from typing import Any

class Prediction_Path():
    _feature_value:str
    _node: Tree_Node|None

    def __init__(self, feature_value:str, node:Tree_Node|None = None) -> None:
        self._feature_value = feature_value
        self._node = node

    @property
    def feature_value(self) -> str:
        """Returns the feature value that is assocatied with this path"""
        return self._feature_value

    @property
    def node(self) -> Tree_Node|None:
        """Returns the next Tree_Node that is reached by following this path"""
        return self._node
        

class Tree_Node():
    _children: tuple(Prediction_Path)|None #type:ignore
    _predict: str|None
    _feature: str|None

    def __init__(self, children:tuple(Prediction_Path)|None = None, predict:str|None = None, feature:str|None = None) -> None: #type:ignore
        self._children = children 
        self._predict = predict
        self._feature = feature

    @property
    def children(self) -> tuple(Prediction_Path) | None: #type:ignore
        """Returns the tuple containing the child nodes.
        Set to None if it is a leaf node.
        """
        return self._children
    
    @property
    def predict(self) -> str | None: 
        """The value that would be predicted if a entry traversed to this leaf node.
        Set to None if it is an internal node.
        """
        return self._predict
    
    @property
    def feature(self) -> str | None: 
        """The feature value that would be split on if a entry traversed to this internal node.
        Set to None if it is a leaf node.
        """
        return self._feature

class Decision_Tree():
    _root: Tree_Node

    def __init__(self) -> None:
        _root = None

    @property
    def root(self) -> Tree_Node|None: #type:ignore
        return self._root
        
    def build_decision_tree(self, data_frame: pd.DataFrame, num_levels: int) -> Tree_Node:
        """Takes a formated dataframe (lable column in the right most position) and builds a decision tree that is num_levels tall.
        The data frame must have at least n features where n = num_levels
        
        For example if num_levles is 1 then the resuting deciston tree has only a root node and leaf nodes."""

        if num_levels == 0:
            values = data_frame[data_frame.columns[-1]].unique()
            value_counts: list[int] = []
            for value in values:
                value_counts.append(len(data_frame[data_frame[data_frame.columns[0]]==value]))
            max_value_index: int = 0
            for _ in range(len(value_counts)):
                if value_counts[_] > value_counts[max_value_index]:
                    max_value_index = _
            return Tree_Node(predict=values[max_value_index])

        #checks that there are still muiltiple lables
        values = data_frame[data_frame.columns[-1]].unique()
        value_counts: list[int] = []
        for value in values:
            value_counts.append(len(data_frame[data_frame[data_frame.columns[0]]==value]))
        max_value_index: int = 0
        num_none_zero_counts: int = 0
        for _ in range(len(value_counts)):
            if value_counts[_] > 0:
                num_none_zero_counts += 1
            if value_counts[_] > value_counts[max_value_index]:
                max_value_index = _
        if (num_none_zero_counts>1):
            return Tree_Node(predict=values[max_value_index])

        feature: str = find_best_info_gain(data_frame)[0]
        child_data_frames: list[pd.DataFrame] = split_on_feature(data_frame,feature)
        prediction_paths: list[Prediction_Path] =[]

        for child in child_data_frames:
            prediction_paths.append(Prediction_Path(child[feature].iloc[0], self.build_decision_tree(child.drop(columns=feature), num_levels-1)))

        self._root = Tree_Node(children=tuple(prediction_paths), feature=feature)
        return self.root #type:ignore
            
    def predict_df(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Takes a formated dataframe (define formated) and mutates it by adding a column on the end named "Prediction" with each row assigned its predicted lable"""
        data_frame["Prediction"] = ""
        for _ in range(len(data_frame)):
            current_node:Tree_Node = self.root #type:ignore
            while current_node.predict == None:
                for path in current_node.children: #type:ignore
                    if path.feature_value == data_frame.at[_,current_node.feature]: 
                        current_node = path.node
                        break
            data_frame.at[_,"Prediction"] = current_node.predict
        return data_frame

    

def entropy(classes: list[int]) -> float:
    """
    Calculate entropy from the counts of examples belonging to
    each class in a classification problem.
    """
    total_values: int = sum(classes)
    total_entropy: float = 0
    
    for value in classes:
        if value != 0:
            proportion: float = value/total_values
            total_entropy = (proportion*(math.log2(proportion))) + total_entropy

    return -1*(total_entropy)
    

def info_gain(par_e, children_e, children_p) -> float:
    """
    Calculates the information gain for a particular split in a decision tree

    par_e is the entropy of the parent node
    children_e is a pandas series of child node entropies
    children_p is a pandas series of the proportion of examples that filtered to each child node (should be in the same order as children_e)

    """
    children_ep: pd.Series = children_e * children_p  
    return float(par_e - children_ep.sum())


def find_best_info_gain(df:pd.DataFrame) -> tuple[str,float]: 
    """ Takes in a dataframe that has its lable in the right most column
    and then calculates the information gain for each column
    and returns the column name and the info gain of the largest info gain
    """
    classes: list = df[df.columns[-1]].unique().tolist()
    num_for_each_class:list = []
    for value in classes:
        num_for_each_class.append(len(df[df[df.columns[-1]] == value]))
    
    entropy_curr: float = entropy(num_for_each_class)
    best_info_gain: float = -1.0
    best_column: str = ""
    for feature in df.columns[0:-1]:
        children_entropy: list = []
        children_proportion: list = []
        options: list = df[feature].unique().tolist()
        
        for option in options:
            df_child: pd.DataFrame = df[df[feature] == option]
            children_proportion.append(len(df_child)/len(df))
            
            num_for_each_child_class:list = []
            for value in classes:
                    num_for_each_child_class.append(len(df_child[df_child[df_child.columns[-1]] == value]))
            children_entropy.append(entropy(num_for_each_child_class))
        
        child_info_gain: float = info_gain(entropy_curr, pd.Series(children_entropy),pd.Series(children_proportion))
        
        if(child_info_gain > best_info_gain):
            best_column = feature
            best_info_gain = child_info_gain
    return (best_column, best_info_gain)


def split_on_feature(df: pd.DataFrame, feature: str) -> list[pd.DataFrame]:
    """Takes a dataframe and a feature and splits the dataframe into n dataframes
    where n is equall to the number of unique values in the feature
    returns a list of n dataframes
    """
    df_list: list = []
    options: list = df[feature].unique().tolist()
    for option in options:
        df_list.append(df[df[feature] == option])
    return df_list

