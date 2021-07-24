import numpy as np
import tensorflow.keras as keras
from scipy import stats
from scipy.spatial.distance import cosine


class clseacher():
    def __init__(self, model:keras.Sequential, input_shape: tuple, metric:keras.metrics.Metric, lang:str='ru') -> None:
        self.input_shape = input_shape
        self.monitor_metric = metric
        self.track_metric = []
        self.steps = []
        self.baseline, self.seacher_idx, self.compiled_options = self._get_baseline(model)
        self.current_arch = self.baseline.copy()
        if lang == "ru":
            self._strings = [
                "Неправильная размерность аргумента data! Должна быть 4, передано: ",
                "Поиск закончен на 0 шаге, базовая архитектура не соответствует критерию метрики.",
                "{:10s} | {:18s} | {:22s} | {:20s} |".format("Итерация", "Совокупно отсеяно", "Количество параметров", "Метрика"),
                "Поиск завершен, метрика не удовлетворяет критерию А:",
                "Поиск завершен, минимальное количество фильтров"
            ]
            
        else:
            self._strings = [
                "Invalid dimension for data argument! Should be 4, passed: ",
                "The search is completed at step 0, the basic architecture does not achieve the metric criterion.",
                "{:10s} | {:18s} | {:22s} | {:20s} |".format("Iteration", "Discarded", "Number of parameters", "Metric"),
                "The search is completed, the metric does not achieve the criterion A:",
                "Search completed, minimum number of filters"
            ]
        
    def _get_baseline(self, model:keras.Sequential) -> tuple:
        baseline_search = []
        seacher_idx = []
        step = {}
        cl_counter = 1
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                baseline_search.append((layer.__class__, layer.get_config()))
                step[f"cl_{cl_counter}"] = baseline_search[-1][1]["filters"]
                seacher_idx.append(idx)
                cl_counter += 1
            else:
                baseline_search.append((layer.__class__, layer.get_config()))
        
        self.steps.append(step)
        compiled_options = {
            "loss" : model.loss, 
            "optimizer_class" : model.optimizer.__class__,
            "optimizer_config": model.optimizer.get_config()
        }
        return baseline_search, seacher_idx, compiled_options
    
    def _build_model(self, changes:list) -> keras.Sequential:
        
        if len(changes) != 0:
            for new_param, old_param in enumerate(self.seacher_idx):
                self.current_arch[old_param][1]["filters"] = changes[new_param]
        
        new_model = [keras.layers.Input(shape=self.input_shape)]
        for item in self.current_arch:
            new_model.append(item[0](**item[1]))

        new_model = keras.Sequential(new_model, name="temp_model")
        return new_model
    
    def _check_normal_layer(self, weights: np.array, pvalue:float) -> list:
        if sum(weights.shape[:-1]) < 10:
            return []

        init_num_filters = weights.shape[-1]
        drop = []
        for w in range(init_num_filters):
            _, p = stats.normaltest(weights[:,:,:,w].flatten())
            if p > pvalue:
                drop.append(w)
            else:
                continue

        return drop
    
    def _cosine_test_layer(self, weights: np.array, treshold=0.1) -> tuple:
        confusion = np.eye(weights.shape[-1])
        for i in range(weights.shape[-1]):
            for j in range(weights.shape[-1]):
                if i != j:
                    confusion[i, j] = cosine(weights[:,:,:,i].flatten(), weights[:,:,:,j].flatten())
                else:
                    continue

        new_confusion = np.delete(confusion, np.where(confusion < treshold)[0], axis=0)
        return np.where(confusion < treshold)[0], confusion, new_confusion
    
    def _model_check_cl_layers(self, model:keras.Sequential, pvalue=0.005, treshold=0.1) -> list:
        
        changes = []
        layers = []
        for idx, item in enumerate(model.layers):
            if idx in self.seacher_idx:
                layers.append(item)
        
        step = {}
        for idx, item in enumerate(layers, start=1):
            first_drop = self._check_normal_layer(item.get_weights()[0], pvalue=pvalue)
            second_drop, _, _ = self._cosine_test_layer(item.get_weights()[0], treshold=treshold)
            all_drop = set(first_drop).union(set(second_drop))
            changes.append(item.get_config()["filters"] - len(all_drop))
            step[f"cl_{idx}"] = changes[-1]
        self.steps.append(step)
            
        return changes
    
    def search(self, data:tuple, epochs:int, batch_size:int, save_path_folder:str, eq_type:bool=True, A:float=0.999, alpha:float=0.1, p_value:float=0.005, print_iter:bool=True):
        
        keras.backend.clear_session()
        model = self._build_model([])
        model.compile(
            loss=self.compiled_options["loss"],
            optimizer=self.compiled_options["optimizer_class"](**self.compiled_options["optimizer_config"]),
            metrics=[self.monitor_metric()]
        )
        
        assert len(data) == 4, self._strings[0] + str(len(data))
        model.fit(data[0], data[1], batch_size=batch_size, epochs=epochs, verbose=0)
        _, metric = model.evaluate(data[2], data[3], batch_size=batch_size, verbose=0)
        self.track_metric.append(metric)
        
        if save_path_folder[-1] not in "/\\":
            save_path_folder += "/"
        model.save(f'{save_path_folder}baseline.h5')
        
        if eq_type:
            if metric < A:
                print(self._strings[1] + " A: " + str(A) + " > " + str(round(metric, 5)))
                return
        else:
            if metric > A:
                print(self._strings[1] + " A: " + str(A) + " < " + str(round(metric, 5)))
                return
        
        if print_iter:
            print(self._strings[2])
        
        changes = self._model_check_cl_layers(model, pvalue=p_value, treshold=alpha)
        
        iteration = 1
        while True:
            keras.backend.clear_session()
            model = self._build_model(changes)
            model.compile(
                loss=self.compiled_options["loss"],
                optimizer=self.compiled_options["optimizer_class"](**self.compiled_options["optimizer_config"]),
                metrics=[self.monitor_metric()]
            )
            model.fit(data[0], data[1], batch_size=batch_size, epochs=epochs, verbose=0)
            _, metric = model.evaluate(data[2], data[3], batch_size=batch_size, verbose=0)
            self.track_metric.append(metric)
            model.save(f'{save_path_folder}iteration_{iteration}.h5')
            
            if print_iter:
                print("{:10s} | {:18s} | {:22s} | {:20s} |".format(str(iteration), str(sum(changes)), str(model.count_params()), str(round(metric, 6))))
            
            if eq_type:
                if metric < A:
                    print(self._strings[3] + " " + str(A) + " > " + str(round(metric, 6)))
                    return
            else:
                if metric > A:
                    print(self._strings[3] + " " + str(A) + " < " + str(round(metric, 6)))
                    return
                
            if sum(changes) == 0:
                print(self._strings[4])
                return
                
            changes = self._model_check_cl_layers(model, pvalue=p_value, treshold=alpha)
            iteration += 1
