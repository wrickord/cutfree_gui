# Standard library imports
import os


class SaveDirectory:
    def __init__(self, model_type=None):
        cur_dir = "./"
        if model_type is None:
            model_type = "exp"
        
        path_name = f"{cur_dir}/_exps_/"
        if not os.path.exists(path_name):
            os.mkdir(path_name)

        mk_dir = True
        current_exp = 1
        while os.path.exists(f"{path_name}/{model_type}-{current_exp}"):
            if not os.listdir(f"{path_name}/{model_type}-{current_exp}"):
                mk_dir = False
                break
            current_exp += 1

        self.save_dir = f"{path_name}/{model_type}-{current_exp}"
        if mk_dir: 
            os.mkdir(self.save_dir)

    def get_dir(self):
        return self.save_dir