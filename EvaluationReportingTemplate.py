"""
This Class provide encapsulation that will show evaluation based on various categories and  metrics , e.g
--------------------------------------------------------------------------------------------------------------------------------------------------------
| Type                     | Control/Predicted   |avg_IOU              | avg_psnr_inter       | avg_psnr_union       | avg_coverage        | Count     |
--------------------------------------------------------------------------------------------------------------------------------------------------------
| HCHI                     | control             |0.241                | 10.098               | 5.366                | 0.044               | 7973      |
| HCHI                     | predicted           |0.407                | 9.357                | 0.823                | 0.058               | 7973      |
| HCLI                     | control             |0.027                | 7.528                | 6.747                | 0.018               | 990       |
| HCLI                     | predicted           |0.259                | 8.188                | 0.845                | 0.04                | 990       |
| LCHI                     | control             |0.125                | 13.214               | 3.799                | 0.004               | 82        |
| LCHI                     | predicted           |0.319                | 10.717               | 0.96                 | 0.016               | 82        |
| LCLI                     | control             |0.011                | 8.75                 | 5.576                | 0.004               | 23        |
| LCLI                     | predicted           |0.15                 | 8.329                | 0.852                | 0.06                | 23        |
--------------------------------------------------------------------------------------------------------------------------------------------------------
| Total                    | control             |0.216                | 9.842                | 5.503                | 0.041               | 9068      |
| Total                    | predicted           |0.389                | 9.239                | 0.826                | 0.056               | 9068      |
--------------------------------------------------------------------------------------------------------------------------------------------------------
Created on Sun Jul 26 11:17:09 2020

@author:  mukul badhan
on Sun Jul 23 11:17:09 2022
"""

import logging

class EvaluatationReporting_reg:
    def __init__(self):
        self.evals = {
            'control': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {},
                'avg_coverage': {}
            },
            'predicted': {
                'avg_IOU': {},
                'avg_psnr_inter': {},
                'avg_psnr_union': {},
                'avg_coverage': {}
            },
            'typecount': {}
        }


    def update(self, eval_single):
        
        type = eval_single.type
        
        control_metric = self.evals['control']
        control_metric['avg_IOU'][type] = control_metric['avg_IOU'].get(type, 0.0) + eval_single.IOU_control
        control_metric['avg_psnr_inter'][type] = control_metric['avg_psnr_inter'].get(type, 0.0) + eval_single.psnr_control_inter
        control_metric['avg_psnr_union'][type] = control_metric['avg_psnr_union'].get(type, 0.0) + eval_single.psnr_control_union
        control_metric['avg_coverage'][type] = control_metric['avg_coverage'].get(type, 0.0) + eval_single.coverage_control
        
        predicted_metric = self.evals['predicted']
        predicted_metric['avg_IOU'][type] = predicted_metric['avg_IOU'].get(type, 0.0) + eval_single.IOU_predicted
        predicted_metric['avg_psnr_inter'][type] = predicted_metric['avg_psnr_inter'].get(type, 0.0) + eval_single.psnr_predicted_inter
        predicted_metric['avg_psnr_union'][type] = predicted_metric['avg_psnr_union'].get(type, 0.0) + eval_single.psnr_predicted_union
        predicted_metric['avg_coverage'][type] = predicted_metric['avg_coverage'].get(type, 0.0) + eval_single.coverage_predicted
        
        # predicted_metric['avg_acuracy'][type] = predicted_metric['avg_acuracy'].get(type, 0.0) + eval_single.coverage_converage
        self.evals['typecount'][type] = self.evals['typecount'].get(type, 0) + 1

    
    def get_evaluations(self):
        return self.evals
    
    def report_results(self):
        # Get the dynamic metric keys from the first category (e.g., 'control')
        metric_keys = list(self.evals['control'].keys())
        
        # Build the header line based on metric keys
        header_row = "| {:<25}| {:<20}|".format("Type", "Control/Predicted") + " | ".join([f"{key:<20}" for key in metric_keys]) + "| {:<10}|".format("Count")
        
        # header_line = '-' * (26 + 22 * len(metric_keys) + 12)
        header_line = '-' * (len(header_row))
        logging.info(header_line)
        logging.info(header_row)
        logging.info(header_line)
        
        for type_name in self.evals['typecount']:
            typecount = self.evals['typecount'][type_name]
            self._log_type_evaluation(type_name, "control", typecount, metric_keys)
            self._log_type_evaluation(type_name, "predicted", typecount, metric_keys)

        logging.info(header_line)
        self._log_total_evaluation("control", metric_keys)
        self._log_total_evaluation("predicted", metric_keys)
        logging.info(header_line)


    def _log_type_evaluation(self, type_name, category, typecount, metric_keys):
        values = [self.evals[category][key].get(type_name , 0.0) / (typecount if key.startswith('avg') else 1)  for key in metric_keys]

        logging.info("| {:<25}| {:<20}|".format(type_name, category) + " | ".join([f"{round(value,3):<20}" for value in values]) + "| {:<10}|".format(typecount))

    def _log_total_evaluation(self, category, metric_keys):
        count = sum(self.evals['typecount'].values())
        avg_values = [sum(self.evals[category][key].values()) / count for key in metric_keys]

        logging.info("| {:<25}| {:<20}|".format("Total", category) + " | ".join([f"{round(value,3):<20}" for value in avg_values]) + "| {:<10}|".format(count))
        