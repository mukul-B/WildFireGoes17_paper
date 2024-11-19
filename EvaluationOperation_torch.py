# -*- coding: utf-8 -*-
"""
This script will run for single image, get all the evaluation Metrics and encasulate result for reporting, and plot the mentioned Results

Created on Sun Jul 26 11:17:09 2020

@author:  mukul badhan
on Sun Jul 23 11:17:09 2022
"""
import logging

# import matplotlib.pyplot as plt
import numpy as np
import torch
from EvaluationMetricsAndUtilities_torch import get_IOU, PSNR_intersection, PSNR_union, best_threshold_iteration, denoralize, get_coverage, getth, noralize_goes_to_radiance, noralize_viirs_to_radiance, psnr
from PlotInputandResults import ImagePlot, plot_from_ImagePlot
from EvaluationMetricsAndUtilities_numpy import getth as getth2

Prediction_JACCARD_LABEL = 'Prediction(Jaccard)'
Prediction_RMSE_LABEL = 'Prediction'
VIIRS_GROUND_TRUTH_LABEL = 'VIIRS Ground Truth'
OTSU_thresholding_on_GOES_LABEL = 'OTSU thresholding on GOES'
GOES_input_LABEL = 'GOES Band 7'
Prediction_Segmentation_label = "Prediction: Segmentation"
Prediction_Regression_label = "Prediction: Regression"
Prediction_RegressionWtMask_label = "Prediction: Regression wt mask"
Prediction_Classification_label = "Prediction: Classification"

# plt.style.use('plot_style/wrf')
from GlobalValues import ALL_SAMPLES, GOES_MAX_VAL, GOES_MIN_VAL, GOES_UNITS, HC, HI, LI, LC, SELECTED_SAMPLES, THRESHOLD_COVERAGE, THRESHOLD_IOU, VIIRS_MAX_VAL, VIIRS_MIN_VAL, VIIRS_UNITS, GOES_Bands

class EvaluationVariables:
    def __init__(self,type):
        self.type = type
        self.iteration, self.ret, self.th = 0, 0, None
        self.th_l1 = None
        self.th_img = None
        self.iou = 0
        self.psnr_intersection = 0
        self.psnr_union = 0
        self.coverage = 0
        self.imagesize = 0
        self.dis = None

class EvaluateSingle:
    def __init__(self,coverage_control,IOU_control, psnr_control_inter, psnr_control_union,coverage_predicted, IOU_predicted, psnr_predicted_inter, psnr_predicted_union, type):
        
        
        self.IOU_control = IOU_control
        self.psnr_control_inter = psnr_control_inter
        self.psnr_control_union = psnr_control_union
        self.coverage_control = coverage_control
        
        self.IOU_predicted = IOU_predicted
        self.psnr_predicted_inter = psnr_predicted_inter
        self.psnr_predicted_union = psnr_predicted_union
        self.coverage_predicted = coverage_predicted
        self.type = type
        
def get_evaluation_results(prediction_rmse, prediction_IOU, inp, groundTruth, path, batch_idx, in_filename, LOSS_NAME,model_name = None, frp = None):
    
    # move from evaluation

    # 1) Evaluation on Input after OTSU thresholding
    inputEV = EvaluationVariables("input")
    extract_img = inp if GOES_Bands == 1 else inp[:, 0:1, :, :]
    inputEV.iteration, inputEV.ret, inputEV.th = best_threshold_iteration(groundTruth, extract_img)
    _, inputEV.th_l1, _, _, _ = getth(extract_img, on=0)
    inputEV.th_img = inputEV.th * extract_img
    inputEV.iou = get_IOU(groundTruth, inputEV.th)
    inputEV.psnr_intersection = PSNR_intersection(groundTruth, inputEV.th_img,max_val=1.0)
    inputEV.psnr_union = PSNR_union(groundTruth, inputEV.th_img,1)
    inputEV.coverage = get_coverage(inputEV.th_l1)
    inputEV.dis = f'\nThreshold (Iteration:{str(inputEV.iteration)}): {str(round(inputEV.ret, 4))} Coverage: {str(round(inputEV.coverage, 4))}' \
                f'\nIOU : {str(inputEV.iou)}' \
                f'\nPSNR_intersection : {str(round(inputEV.psnr_intersection, 4))}'

     #  2) rmse prediction evaluation
    binary_prediction = True 

    # finding coverage and intensity criteria based on input and groundthruth
    condition_coverage = HC if inputEV.coverage > THRESHOLD_COVERAGE else LC
    condition_intensity = HI if inputEV.iou > THRESHOLD_IOU else LI
    condition = condition_coverage + condition_intensity

    predRMSEEV = EvaluationVariables("prediction_rmse")
    if prediction_rmse is not None:
        binary_prediction = False
        outmap_min = prediction_rmse.min()
        outmap_max = prediction_rmse.max()
        prediction_rmse_normal = (prediction_rmse - outmap_min) / (outmap_max - outmap_min)

        prediction_rmse = prediction_rmse_normal
        prediction_rmse = torch.nan_to_num(prediction_rmse)
        
        predRMSEEV.ret, predRMSEEV.th, histogram,_, _ = getth(prediction_rmse, on=0)
        
        predRMSEEV.th_img = predRMSEEV.th * prediction_rmse
        predRMSEEV.iou = get_IOU(groundTruth, predRMSEEV.th)
        
        predRMSEEV.psnr_intersection = PSNR_intersection(groundTruth, predRMSEEV.th_img,1)
        predRMSEEV.psnr_union = PSNR_union(groundTruth, predRMSEEV.th_img,1)
        predRMSEEV.coverage = get_coverage(predRMSEEV.th)
        # (torch.count_nonzero(predRMSEEV.th) / predRMSEEV.th.numel()).item()
        
        predRMSEEV.dis = f'\nThreshold: {str(round(predRMSEEV.ret, 4))}  Coverage:  {str(round(predRMSEEV.coverage, 4))} ' \
                f'\nIOU :  {str(predRMSEEV.iou)} ' \
                f'\nPSNR_intersection : {str(round(predRMSEEV.psnr_intersection, 4))}'
        prediction_rmse = predRMSEEV.th_img
        Prediction_LABEL = Prediction_Regression_label +' '+ model_name

    # 5)IOU prediction evaluation
    predIOUEV = EvaluationVariables("prediction_jaccard")
    if prediction_IOU is not None:
        

        predIOUEV.ret, predIOUEV.th,  _,_, _ = getth(prediction_IOU, on=0)
        predIOUEV.th_img = predIOUEV.th * prediction_IOU
        predIOUEV.iou = get_IOU(groundTruth, predIOUEV.th)
        predIOUEV.coverage = get_coverage(predIOUEV.th)
        predIOUEV.dis = f'\nThreshold:  {str(round(predIOUEV.ret, 4))}  Coverage:  {str(round(predIOUEV.coverage, 4))} ' \
                f'\nIOU :  {str(predIOUEV.iou)}'
        prediction_IOU = predIOUEV.th_img

    (coverage_p,iou_p, psnr_intersection_p, psnr_union_p) = (predRMSEEV.coverage,predRMSEEV.iou, predRMSEEV.psnr_intersection, predRMSEEV.psnr_union) if (
            prediction_rmse is not None) else (predIOUEV.coverage,predIOUEV.iou, 0, 0)
    
    
    eval_single = EvaluateSingle(inputEV.coverage,
                                 inputEV.iou,
                                 inputEV.psnr_intersection,
                                 inputEV.psnr_union,
                                 coverage_p,
                                 iou_p,
                                 psnr_intersection_p,
                                 psnr_union_p,
                                 condition) 
    
    
    site_date_time = '_'.join(in_filename.split('.')[1].split('_')).replace(' ','_' ).replace('-','_' )
    

    if ALL_SAMPLES or str(batch_idx) in SELECTED_SAMPLES :
        
        inp,groundTruth,prediction_rmse,prediction_IOU = move_to_CPU(inp,groundTruth,prediction_rmse,prediction_IOU)
        
        
        inp = inp.numpy()
        groundTruth = groundTruth.numpy()
        frp = frp.numpy()
        prediction_rmse = prediction_rmse.numpy() if prediction_rmse is not None else None
        prediction_IOU = prediction_IOU.numpy() if prediction_IOU is not None else None
        shape_result = groundTruth.shape
        
        # 1) Plot Input ---------------------------------------------------------------------------------------
        
        extract_img = inp if GOES_Bands == 1 else inp[0]
        g1 = ImagePlot(GOES_UNITS,GOES_MAX_VAL[0], GOES_MIN_VAL[0],
                       extract_img, 
                       GOES_input_LABEL)
        

        # 2)Plot Ground truth ---------------------------------------------------------------------------------------
        
        viirs_30wl = np.sum((frp > 0)& (frp <=30))
        viirs_30wm = np.sum(frp >30)
        viirs30wratio = viirs_30wm /viirs_30wl

        g3 = ImagePlot(VIIRS_UNITS,VIIRS_MAX_VAL,VIIRS_MIN_VAL,
                       groundTruth ,
                       VIIRS_GROUND_TRUTH_LABEL +'_' + str(round(inputEV.coverage,3)) +'_'+ str(round(inputEV.iou,3)) )


        # 3)Plot Prediction ---------------------------------------------------------------------------------------
        
        extra_label  = f':\n{str(round(coverage_p,3))}_{str(round(iou_p,3))}_{str(round(psnr_intersection_p,2))}'
        g4 = ImagePlot(VIIRS_UNITS if prediction_rmse is not None else "IOU",VIIRS_MAX_VAL,VIIRS_MIN_VAL,
                    prediction_rmse if prediction_rmse is not None else prediction_IOU,
                    Prediction_LABEL + extra_label if prediction_rmse is not None else Prediction_JACCARD_LABEL,binary_prediction)

        

        img_seq = ((g1,g3,g4),)
        
        
        name_date_split = site_date_time.split('_')
        yy,mm,dd,hm = name_date_split[-4:]
        site_name = ' '.join(name_date_split[:-4])

        title_plot = f'{site_name} {yy}-{mm}-{dd} {hm}'
        
        out_filename = f'{iou_p:.3f}_{batch_idx}'
        path = f'{path}/{condition}/{out_filename}.png'
        
        plot_from_ImagePlot(title_plot,img_seq,path,shape=shape_result,colection=True)
        
        logging.info(
        f'{LOSS_NAME},{condition},{batch_idx},{str(predRMSEEV.iou) if prediction_rmse is not None else ""},{str(predIOUEV.iou) if prediction_IOU is not None else ""},{predRMSEEV.psnr_intersection if prediction_rmse is not None else ""},{site_date_time}')
        
        
    return eval_single


def move_to_CPU(x,y,output_rmse,output_jaccard):
    x = x.cpu()
    y = y.cpu()
    x = np.squeeze(x)
    y = np.squeeze(y)

    if output_rmse is not None:
        # output_rmse = output_rmse.view(1, 128, 128)
        output_rmse = output_rmse.cpu()
        output_rmse = np.squeeze(output_rmse)
    if output_jaccard is not None:
        output_jaccard = output_jaccard.cpu()
        output_jaccard = np.squeeze(output_jaccard)
    nonzero = np.count_nonzero(output_rmse)
    return x,y,output_rmse,output_jaccard
