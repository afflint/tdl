import pandas as pd 
import ast


class ItalianRecipes:
    
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path, index_col=0)
    
    @property
    def ingredients(self):
        ingredients = [ast.literal_eval(x) for x in self.df['Ingredienti']]
        return ingredients
    
    @property
    def titles(self):
        return list(self.df['Nome'])
    
    def ingredients_text(self, include_title: bool = False):
        docs = []
        titles = self.titles
        for i, ingredient_list in enumerate(self.ingredients):
            ingredients_text = " ".join([x[0].replace(" ", "_") for x in ingredient_list])
            if include_title:
                ingredients_text = " ".join([titles[i].replace(" ", "_"), ingredients_text])
            docs.append(ingredients_text)
        return docs