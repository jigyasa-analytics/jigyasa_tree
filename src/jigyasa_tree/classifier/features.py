from sklearn.tree import DecisionTreeClassifier, _tree
import pandas as pd

def add_node_variables(model, dataset, feature_names):    

    tree=model.tree_

    df=dataset.copy()

    node_rules={}

    def traverse(node=0, conditions=None):

        if node==-1:
            return
        
        if conditions is None:
            conditions=[]

        node_rules[node]=conditions.copy()

        if tree.feature[node]==_tree.TREE_UNDEFINED:
            return

        feature=feature_names[tree.feature[node]]
        threshold=tree.threshold[node]

        left_conditions= conditions + [ (feature, "<=", threshold)]
        traverse(tree.children_left[node], left_conditions)

        right_conditions= conditions + [ (feature, ">", threshold)]
        traverse(tree.children_right[node], right_conditions)
    
    traverse()       
    node_vars=[]
    rule_strings=[]
    pref="NO"
    
    for node, conditions in node_rules.items():       
           
        col=f"{pref}{node}"
        node_vars.append(col)
            
        if len(conditions)==0:
            df[col]=1

            rule_text=f"df['{col}']=1"
            rule_strings.append(rule_text)
            continue            

        mask=pd.Series(True, index=df.index)
            
        cond_texts=[]
            
        for feature, operator, threshold in conditions:

            if operator == "<=":
                 mask&=df[feature] <= threshold

            else:
                mask&=df[feature] > threshold
                    
            cond_texts.append(f"(df['{feature}']{operator}{threshold})")

        df[col]=mask.astype(int)
        combined_conditions="&".join(cond_texts)
        rule_text=(f"df.loc[{combined_conditions},'{col}']=1")            
        rule_strings.append(rule_text)

    return df, rule_strings
 
