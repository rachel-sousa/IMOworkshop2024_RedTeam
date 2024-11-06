# ====================================================================================
# Various functions that I found useful in this project
# ====================================================================================
import re
import numpy as np
import pandas as pd
import sys
import os
import pickle
import scipy
from scipy.stats import t
from lmfit import minimize
from tqdm import tqdm
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sys.path.append("./")
import myUtils as utils
from odeModels import create_model

# ====================================================================================
def residual(params, x, data, model, model_to_observation_map, solver_kws={}, residual_kws={}):
    time_step_adjust_up = residual_kws.get('time_step_adjust_up',1.25)
    time_step_adjust_down = residual_kws.get('time_step_adjust_down',0.75)
    begin_stabilisation_threshold = residual_kws.get('begin_stabilisation_threshold',1e-1)
    verbose = residual_kws.get('verbose',False)
    # Set initial conditions
    for var in model.stateVars:
        if var in model_to_observation_map.keys():
            params[var+'0'].value = data[model_to_observation_map[var]].iloc[0]
        else:
            params[var+'0'].value = 0
    model.SetParams(**params.valuesdict())
    converged = False
    max_step = min(time_step_adjust_up*model.max_step, solver_kws.get('max_step',np.inf))
    currSolver_kws = solver_kws.copy()
    currSolver_kws['numericalStabilisationB'] = (max_step < begin_stabilisation_threshold*solver_kws.get('max_step',np.inf)) or model.numericalStabilisationB
    while not converged:
        currSolver_kws['max_step'] = max_step
        model.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(data), **currSolver_kws)
        converged = model.successB
        if verbose and (not converged or currSolver_kws['numericalStabilisationB']): print([max_step, currSolver_kws['numericalStabilisationB'], np.any(model.solObj.y<0), model.solObj.status, model.solObj.success, model.errMessage])
        max_step = time_step_adjust_down*max_step if max_step < np.inf else 100
        currSolver_kws['numericalStabilisationB'] = max_step < begin_stabilisation_threshold*solver_kws.get('max_step',np.inf)
    # Turn off numerical stabilisation once we've been able to increase the time step above the threshold again
    if not currSolver_kws['numericalStabilisationB']: model.numericalStabilisationB = False
    # Interpolate to the data time grid
    t_eval = data.Time
    tmp_list = []
    for model_feature in model_to_observation_map:
        observed_feature = model_to_observation_map[model_feature]
        f = scipy.interpolate.interp1d(model.resultsDf.Time,model.resultsDf[model_feature],fill_value="extrapolate")
        modelPrediction = f(t_eval)
        scale_dict = residual_kws.get('residual_scale',{model_feature:1.})
        res_list = (data[observed_feature]-modelPrediction) / scale_dict[model_feature]
        tmp_list.append(res_list)
    return np.concatenate(tmp_list)

# ====================================================================================
def residual_multipleConditions(params, x, data, model, model_to_observation_map, split_by,
                                solver_kws={}, residual_kws={}):
    tmpList = []
    for condition in data[split_by].unique():
        currData = data[data[split_by]==condition]
        tmpList.append(residual(params, x, currData, model, model_to_observation_map, solver_kws=solver_kws, residual_kws=residual_kws.get(condition, {})))
    return np.concatenate(tmpList)

# ====================================================================================
def residual_multipleTxConditions(params, x, data, model, model_to_observation_map, solver_kws={}, residual_kws={}):
    return residual_multipleConditions(params, x, data, model, model_to_observation_map, split_by="DrugConcentration", 
                                       solver_kws=solver_kws, residual_kws=residual_kws)

# ====================================================================================
def PerturbParams(params):
    params = params.copy()
    for p in params.keys():
        currParam = params[p]
        if currParam.vary:
            params[p].value = np.random.uniform(low=currParam.min, high=currParam.max)
    return params

# ====================================================================================
def compute_r_sq(fit,dataDf,feature="Confluence"):
    tss = np.sum(np.square(dataDf[feature]-dataDf[feature].mean()))
    rss = np.sum(np.square(fit.residual))
    return 1-rss/tss

# ====================================================================================
def prepare_data(data_df, specs_dic, restrict_range=True, average=True, average_by="Time"):
    '''
    Prepare data for fitting. This function subsets the data based on the specifications provided in specs_dic.
    :param data_df: Pandas data frame with longitudinal data to be fitted.
    :param specs_dic: Dictionary with specifications for subsetting the data. The keys are the names of the columns
    in the data frame, and the values are the values to be selected.
    :param restrict_range: Boolean, whether or not to restrict the time range to the one specified in specs_dic.
    :param average: Boolean, whether or not to average across replicates.
    :return: Pandas data frame with the data to be fitted.
    '''
    # Prepare data
    cols_available = data_df.columns
    training_data_df = data_df.copy()
    # Subset the data as specified in specs_dic
    for filter_name, filter_value in specs_dic.items():
        # specs_dic might give other details too. Only apply those col names 
        # that actually exist in the dataframe
        if filter_name in cols_available:
            filter_value_list = filter_value if isinstance(filter_value, list) else [filter_value] # Users can specify one or multiple possible values. If only one value is specified, convert to list here
            training_data_df = training_data_df[np.isin(training_data_df[filter_name], filter_value_list)].copy()
    # Limit the time if requested
    if restrict_range and specs_dic.get("TimeRange", [-np.inf])[0]>=0:
        training_data_df['Time_original'] = training_data_df['Time']
        training_range = specs_dic["TimeRange"]
        training_data_df = training_data_df[(training_data_df.Time>=training_range[0]) &
                                            (training_data_df.Time<=training_range[1])].copy()
        training_data_df['Time'] -= training_range[0]
    # Average across replicates
    if average: 
        training_data_df = training_data_df.groupby(by=average_by).mean(numeric_only=True)
        training_data_df.reset_index(inplace=True)
    return training_data_df

# ====================================================================================
def plot_data(dataDf, timeColumn="Time", feature='CA153', 
              treatmentColumn="DrugConcentration", treatment_notation_mode="post",
              estimator=None, n_boot=100, err_style="bars",
              hue=None, style=None, legend=False, palette=None,
              plotDrug=True, plotDrugAsBar=True, drugBarPosition=0.85, drugBarColour="black",
              drugColorMap={"Encorafenib": "blue", "Binimetinib": "green", "Nivolumab": sns.xkcd_rgb["goldenrod"]},
              xlim=None, ylim=None, y2lim=1,
              markInitialSize=False, markPositiveCutOff=False, plotHorizontalLine=False, lineYPos=1, despine=False,
              titleStr="", decorateX=True, decorateY=True, decorateY2=True,
              markerstyle='o', markersize=12, linestyle="None", linecolor='black',
              ax=None, figsize=(10, 8), outName=None, **kwargs):
    '''
    Plot longitudinal treatment data, together with annotations of drug administration and events responsible for
    changes in treatment dosing (e.g. toxicity).
    :param dataDf: Pandas data frame with longitudinal data to be plotted.
    :param timeColumn: Name (str) of the column with the time information.
    :param feature: Name (str) of the column with the metric to be plotted on the y-axis (e.g. PSA, CA125, etc).
    :param treatmentColumn: Name (str) of the column with the information about the dose administered.
    :param plotDrug: Boolean; whether or not to plot the treatment schedule.
    :param plotDrugAsBar: Boolean, whether to plot drug as bar across the top, or as shading underneath plot.
    :param drugBarPosition: Position of the drug bar when plotted across the top.
    :param drugColorMap: Color map for colouring the shading when using different drugs.
    :param lw_events: Line width for vertical event lines.
    :param xlim: x-axis limit.
    :param ylim: y-axis limit.
    :param y2lim: y2-axis limit.
    :param markInitialSize: Boolean, whether or not to draw horizontal line at height of fist data point.
    :param plotHorizontalLine: Boolean, whether or not to draw horizontal line at position specified at lineYPos.
    :param lineYPos: y-position at which to plot horizontal line.
    :param despine: Boolean, whether or not to despine the plot.
    :param titleStr: Title to put on the figure.
    :param decorateX: Boolean, whether or not to add labels and ticks to x-axis.
    :param decorateY: Boolean, whether or not to add labels and ticks to y-axis.
    :param decorateY2: Boolean, whether or not to add labels and ticks to y2-axis.
    :param markersize: Size of markers for feature variable.
    :param linestyle: Feature variable line style.
    :param linecolor: Feature variable line color.
    :param ax: matplotlib axis to plot on. If none provided creates a new figure.
    :param figsize: Tuple, figure dimensions when creating new figure.
    :param outName: Name under which to save figure.
    :param kwargs: Other kwargs to pass to plotting functions.
    :return:
    '''
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Plot the data
    if style==None: # If no style is specified, use the "marker" keyword. Otherwise points won't show up.
        marker_dic = {"marker":markerstyle}
    else:
        marker_dic = {"markers":markerstyle}
    sns.lineplot(x=timeColumn, y=feature, hue=hue, style=style, err_style=err_style,
                 estimator=estimator, n_boot=n_boot,
                 color=linecolor, legend=legend, palette=palette, 
                 markersize=markersize, markeredgewidth=2, **marker_dic, 
                 ax=ax, data=dataDf)

    # Plot the drug concentration
    if plotDrug:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        if hue is not None or style is not None:
            drug_data_df = dataDf.groupby(timeColumn).mean(numeric_only=True).reset_index()
        else:
            drug_data_df = dataDf
        if plotDrugAsBar:
            drugConcentrationVec = utils.TreatmentListToTS(
                treatmentList=utils.ExtractTreatmentFromDf(drug_data_df, timeColumn=timeColumn,
                                                           treatmentColumn=treatmentColumn,
                                                           mode=treatment_notation_mode),
                tVec=drug_data_df[timeColumn])
            drugConcentrationVec[drugConcentrationVec < 0] = 0
            drugConcentrationVec = np.array([x / (np.max(drugConcentrationVec) + 1e-12) for x in drugConcentrationVec])
            drugConcentrationVec = drugConcentrationVec / (1 - drugBarPosition) + drugBarPosition
            ax2.fill_between(drug_data_df[timeColumn], drugBarPosition, drugConcentrationVec,
                             step="post", color=drugBarColour, alpha=1., label="Drug Concentration")
            ax2.axis("off")
        else:
            currDrugBarPosition = drugBarPosition
            drugBarHeight = (1-drugBarPosition)/len(drugColorMap.keys())
            for drug in drugColorMap.keys():
                drugConcentrationVec = utils.TreatmentListToTS(
                    treatmentList=utils.ExtractTreatmentFromDf(drug_data_df, timeColumn=timeColumn,
                                                               treatmentColumn="%s Dose (mg)"%drug,
                                                               mode=treatment_notation_mode),
                    tVec=drug_data_df[timeColumn])
                drugConcentrationVec[drugConcentrationVec < 0] = 0
                # Normalise drug concentration to 0-1 (1=max dose(=initial dose))
                drugConcentrationVec = np.array([x / (np.max(drugConcentrationVec) + 1e-12) for x in drugConcentrationVec])
                # Rescale to make it fit within the bar at the top of the plot
                drugConcentrationVec = drugConcentrationVec * drugBarHeight + currDrugBarPosition
                ax2.fill_between(drug_data_df[timeColumn], currDrugBarPosition, drugConcentrationVec, step="post",
                                 color=drugColorMap[drug], alpha=0.5, label="Drug Concentration")
                ax2.hlines(xmin=drug_data_df[timeColumn].min(), xmax=drug_data_df[timeColumn].max(), 
                          y=currDrugBarPosition, linewidth=3, color="black")
                currDrugBarPosition += drugBarHeight
            # Line at the top of the drug bars
            ax2.hlines(xmin=drug_data_df[timeColumn].min(), xmax=drug_data_df[timeColumn].max(), 
                          y=currDrugBarPosition, linewidth=3, color="black")
        # Format y2 axis
        if y2lim is not None: ax2.set_ylim([0, y2lim])
        ax2.tick_params(labelsize=28)
        if not decorateY2:
            ax2.set_yticklabels("")

    # Format the plot
    if xlim is not None: ax.set_xlim(0, xlim)
    if ylim is not None: ax.set_ylim(0, ylim)
    if despine: sns.despine(ax=ax, trim=True, offset=50)

    # Draw horizontal lines (e.g. initial size)
    if plotHorizontalLine or markInitialSize or markPositiveCutOff:
        xlim = ax.get_xlim()[1]
        if markInitialSize: lineYPos = dataDf.loc[dataDf[timeColumn] == 0, feature]
        if markPositiveCutOff: lineYPos = 0.5 # cut-off value for positive is 0.5 copies/uL
        ax.hlines(xmin=0, xmax=xlim, y=lineYPos, linestyles=':', linewidth=4)

    # Decorate the plot
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(titleStr)
    ax.tick_params(labelsize=28)
    if not decorateX:
        ax.set_xticklabels("")
    if not decorateY:
        ax.set_yticklabels("")
    plt.tight_layout()
    if outName is not None: plt.savefig(outName)

# ====================================================================================
def PlotFit(fitObj, dataDf, model=None, dt=1, linewidth=5, linewidthA=5, titleStr="", legend=True, outName=None, ax=None, solver_kws={}, **kwargs):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if model is None:
        myModel = create_model(fitObj.modelName)
        myModel.SetParams(**fitObj.params.valuesdict())
    else:
        myModel = model
    solver_kws['max_step'] = solver_kws.get('max_step',1) # Use fine-grained time-stepping unless otherwise specified
    myModel.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(dataDf),**solver_kws)
    myModel.Trim(dt=dt)
    myModel.Plot(ymin=0, title=titleStr, linewidth=linewidth, linewidthA=linewidthA, ax=ax, plotLegendB=legend, **kwargs)
    plot_data(dataDf, plotDrugConcentration=False, ax=ax, **kwargs)
    if outName is not None: plt.savefig(outName); plt.close()

# ====================================================================================
def load_fit(modelName, fitId=0, file_name_model=None, fitDir="./", model=None, load_bootstraps=False, file_name_bootstraps=None, **kwargs):
    file_name_model = "fitObj_fit_%d.p"%(fitId) if file_name_model is None else file_name_model
    fitObj = pickle.load(open(os.path.join(fitDir, file_name_model), "rb"))
    myModel = create_model(modelName, **kwargs) if model is None else model
    myModel.SetParams(**fitObj.params.valuesdict())
    if load_bootstraps:
        file_name_bootstraps = "bootstraps_fit_%d.csv"%(fitId) if file_name_bootstraps is None else file_name_bootstraps
        bootstraps_df = pd.read_csv(os.path.join(fitDir, file_name_bootstraps), index_col=0)
        return fitObj, myModel, bootstraps_df
    else:
        return fitObj, myModel

# ====================================================================================
def generate_fitSummaryDf(fitDir="./fits", identifierName=None, identifierId=1, alpha=0.95):
    '''
    Function to generate a summary data frame with the parameter estimates and confidence intervals.
    :param fitDir: Directory with the fit objects.
    :param identifierName: Name of the identifier to be added to the data frame.
    :param identifierId: Value of the identifier to be added to the data frame.
    :param alpha: Confidence interval.
    :return: Pandas data frame with the parameter estimates and confidence intervals.
    '''
    fitIdList = [int(re.findall(r'\d+', x)[0]) for x in os.listdir(fitDir) if
                 x.split("_")[0] == "fitObj"]
    identifierDic = {} if identifierName is None else {identifierName: identifierId}
    tmpDicList = []
    for fitId in fitIdList:
        fitObj = pickle.load(open(os.path.join(fitDir, "fitObj_fit_%d.p"%(fitId)), "rb"))
        tmpDicList.append({**identifierDic, "FitId": fitObj.fitId, "ModelName":fitObj.modelName,
                           "AIC": fitObj.aic, "BIC": fitObj.bic, "RSquared": fitObj.rSq,
                           "Success":fitObj.success, "Message":fitObj.message, "NumericalStabilisation":fitObj.numericalStabilisation,
                           **fitObj.params.valuesdict(),
                           **dict([(x+"_se",fitObj.params[x].stderr) for x in fitObj.params.keys()]),
                           **dict([(x+"_ci",t.ppf((1+alpha)/2.0, fitObj.ndata-fitObj.nvarys)*(fitObj.params[x].stderr if fitObj.params[x].stderr is not None else np.nan)) for x in fitObj.params.keys()])})
    return pd.DataFrame(tmpDicList)

# ====================================================================================
def perform_bootstrap(fitObj, n_bootstraps=5, shuffle_params=True, prior_experiment_df=None, model_kws={},
                      residual_fun=residual,
                      model_to_observation_map={"TumourSize":'Confluence'}, varyICs=None,
                      split_by=None,
                      show_progress=True, plot_bootstraps=False, plot_kws={'ylim':None,'palette':None}, 
                      outName=None, **kwargs):
    '''
    Function to estimate uncertainty in the parameter estimates and model predictions using a
    parametric bootstrapping method. This means, it uses the maximum likelihood estimate (best fit
    based on least squared method) to generate n_bootstrap synthetic data sets (noise is generated
    by drawing from an error distribution N(0,sqrt(ssr/df))). Subsequently it fits to this synthetic
    data to obtain a distribution of parameter estimates (one estimate/prediction
    per synthetic data set).
    '''
    # Initialise
    nvarys = fitObj.nvarys
    residual_variance = np.sum(np.square(fitObj.residual)) / fitObj.nfree
    paramsToEstimateList = [param for param in fitObj.params.keys() if fitObj.params[param].vary]
    n_conditions = 1 if split_by is None else len(fitObj.data[split_by].unique())
    plot_kws['ylim'] = 1.3*np.max([fitObj.data[x].max() for x in model_to_observation_map.values()]) if plot_kws['ylim'] is None else plot_kws['ylim']
    plot_kws['palette'] = sns.color_palette("pastel", n_colors=n_bootstraps) if plot_kws['palette'] is None else plot_kws['palette']
    if plot_bootstraps:
        fig, ax_list = plt.subplots(1, len(model_to_observation_map), figsize=(len(model_to_observation_map)*8, 6))

    # 1. Perform bootstrapping
    parameterEstimatesMat = np.zeros((n_bootstraps, nvarys+1))  # Array to hold parameter estimates for CI estimation
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # i) Generate synthetic data by sampling from the error model (assuming a normal error distribution)
        tmpDataDf = fitObj.data.copy()
        if split_by is None:
            for observed_feature in model_to_observation_map.values():
                tmpDataDf.loc[np.isnan(tmpDataDf[observed_feature])==False, observed_feature] -= fitObj.residual # Obtain the model prediction
                tmpDataDf.loc[np.isnan(tmpDataDf[observed_feature])==False, observed_feature] += np.random.normal(loc=0, scale=np.sqrt(residual_variance),
                                                                            size=fitObj.ndata)
        else:
            curr_condition_index_start = 0
            curr_condition_index_end = 0
            tmp_list = []
            for condition in tmpDataDf[split_by].unique():
                for observed_feature in model_to_observation_map.values():
                    # Extract the residuals for the current data set
                    curr_data_selection = (np.isnan(tmpDataDf[observed_feature])==False) & (tmpDataDf[split_by]==condition)
                    curr_condition_index_end += tmpDataDf[curr_data_selection].shape[0]
                    curr_residual = fitObj.residual[curr_condition_index_start:curr_condition_index_end]
                    tmpDataDf.loc[curr_data_selection, observed_feature] -= curr_residual # Obtain the model prediction
                    tmpDataDf.loc[curr_data_selection, observed_feature] += np.random.normal(loc=0, scale=np.sqrt(residual_variance),
                                                                                size=curr_residual.shape[0])
                    tmp_list.append({"Condition":condition, "Feature":observed_feature, 
                                     "Range":[curr_condition_index_start, curr_condition_index_end]})
                    curr_condition_index_start = curr_condition_index_end
            data_to_residual_index_map = pd.DataFrame(tmp_list)

        # bestFitPrediction = tmpDataDf[feature] - fitObj.residual
        # tmpDataDf[feature] = bestFitPrediction + np.random.normal(loc=0, scale=np.sqrt(residual_variance),
        #                                                                size=fitObj.ndata)
        # ii) Fit to synthetic data
        tmpModel = create_model(fitObj.modelName, **model_kws)
        currParams = fitObj.params.copy()
        # Remove variation in initial synthetic data if not fitting initial conditions;
        # otherwise this will blow up the residual variance as no fit can ever do well on the IC
        areIcsVariedList = [fitObj.params[stateVar+'0'].vary for stateVar in tmpModel.stateVars]
        if (not np.any(areIcsVariedList)) or (varyICs==False): # Allow manual overwrite via varyICs, in case of only varying some of the ICs
            for observed_feature in model_to_observation_map.values():
                if n_conditions==1:
                    tmpDataDf.loc[0, observed_feature] = fitObj.data[observed_feature].iloc[0]
                else: # If fitting to multiple experiments simultaneously, remove variation from each experiment separately
                    tmpDataDf.loc[tmpDataDf.Time==0, observed_feature] = fitObj.data.loc[fitObj.data.Time==0, observed_feature].values
        # In developing our model we proceed in a series of steps. To propagate the error along
        # as we advance to the next step, allow reading in previous bootstraps here.
        if prior_experiment_df is not None:
            for var in prior_experiment_df.columns:
                if var == "SSR": continue
                currParams[var].value = prior_experiment_df[var].iloc[bootstrapId]
        # Generate a random initial parameter guess
        if shuffle_params:
            for param in paramsToEstimateList:
                currParams[param].value = np.random.uniform(low=currParams[param].min,
                                                            high=currParams[param].max)
        # Fit
        currFitObj = minimize(residual_fun, currParams, args=(0, tmpDataDf, tmpModel,
                                                          model_to_observation_map, kwargs.get('solver_kws', {}), 
                                                          kwargs.get('residual_kws', {})),
                              **kwargs.get('optimiser_kws', {}))
        # Record parameter estimates for CI estimation
        for i, param in enumerate(paramsToEstimateList):
            parameterEstimatesMat[bootstrapId, i] = currFitObj.params[param].value
        parameterEstimatesMat[bootstrapId, -1] = np.sum(np.square(currFitObj.residual))

        # Plot the synthetic data and the individual bootstrap fits. This is useful for i) understanding what
        # the method is doing, and ii) debugging.
        if plot_bootstraps:
            # fig, ax_list = plt.subplots(1, len(model_to_observation_map), figsize=(len(model_to_observation_map)*8, 6))
            for i, observed_feature in enumerate(model_to_observation_map.values()):
                ax = ax_list[i] if len(model_to_observation_map)>1 else ax_list
                ax.plot(tmpDataDf.Time, tmpDataDf[observed_feature], linestyle="", marker='o', linewidth=3, color=plot_kws['palette'][bootstrapId])
                if split_by is None:
                    ax.plot(tmpDataDf.Time[np.isnan(tmpDataDf[observed_feature])==False], 
                            tmpDataDf.loc[np.isnan(tmpDataDf[observed_feature])==False,observed_feature]-currFitObj.residual,
                            linewidth=3, linestyle="-", color=plot_kws['palette'][bootstrapId])
                else:
                    for condition in tmpDataDf[split_by].unique():
                        # Extract the residuals for the current data set
                        curr_data_selection = (np.isnan(tmpDataDf[observed_feature])==False) & (tmpDataDf[split_by]==condition)
                        curr_time = tmpDataDf.Time[curr_data_selection]
                        curr_index_range = data_to_residual_index_map[(data_to_residual_index_map.Condition==condition) & 
                                                                      (data_to_residual_index_map.Feature==observed_feature)].Range.values[0]
                        curr_residual = currFitObj.residual[curr_index_range[0]:curr_index_range[1]]
                        curr_model_prediction = tmpDataDf.loc[curr_data_selection,observed_feature]-curr_residual
                        ax.plot(curr_time, curr_model_prediction,
                                linewidth=3, linestyle="-", color=plot_kws['palette'][bootstrapId])

    # Add the maximum likelihood estimate fit to the plot
    if plot_bootstraps:
        for i, observed_feature in enumerate(model_to_observation_map.values()):
            ax = ax_list[i] if len(model_to_observation_map)>1 else ax_list
            if split_by is None:
                bestFitPrediction = fitObj.data.loc[np.isnan(fitObj.data[observed_feature])==False,observed_feature] - fitObj.residual
                ax.plot(fitObj.data.Time[np.isnan(tmpDataDf[observed_feature])==False], bestFitPrediction, linewidth=5, linestyle="-", color='k')
            else:
                for condition in fitObj.data[split_by].unique():
                    curr_data_selection = (np.isnan(fitObj.data[observed_feature])==False) & (fitObj.data[split_by]==condition)
                    curr_time = fitObj.data.Time[curr_data_selection]
                    curr_index_range = data_to_residual_index_map[(data_to_residual_index_map.Condition==condition) & 
                                                                  (data_to_residual_index_map.Feature==observed_feature)].Range.values[0]
                    curr_residual = fitObj.residual[curr_index_range[0]:curr_index_range[1]]
                    bestFitPrediction = fitObj.data.loc[curr_data_selection,observed_feature] - curr_residual
                    ax.plot(curr_time, bestFitPrediction, linewidth=5, linestyle="-", color='k')
            ax.set_ylim(0,plot_kws['ylim'])
            ax.set_title(observed_feature)

    # Return results
    resultsDf = pd.DataFrame(parameterEstimatesMat, columns=paramsToEstimateList+['SSR'])
    if prior_experiment_df is not None: resultsDf = pd.concat([prior_experiment_df.drop('SSR',axis=1), resultsDf], axis=1)
    if outName is not None: resultsDf.to_csv(outName)
    return resultsDf

# ====================================================================================
def compute_confidenceInterval_prediction(fitObj, bootstrapResultsDf, alpha=0.95,
                                          treatmentScheduleList=None, atToProfile=None, at_kws={},
                                          initialConditionsDic=None, model_kws={},
                                          t_eval=None, n_time_steps=100,
                                          show_progress=True, 
                                          returnTrajectories=False, estimate_fractions=False,
                                          **kwargs):
    # Initialise
    if t_eval is None:
        if treatmentScheduleList is None:
            if atToProfile is None:
                currPredictionTimeFrame = [fitObj.data.Time.min(), fitObj.data.Time.max()]
            else:
                currPredictionTimeFrame = [0, at_kws.get('t_end', 20)]
        else:
            currPredictionTimeFrame = [treatmentScheduleList[0][0], treatmentScheduleList[-1][1]]
        t_eval = np.linspace(currPredictionTimeFrame[0], currPredictionTimeFrame[1], n_time_steps) if t_eval is None else t_eval
    n_timePoints = len(t_eval)
    n_stateVars = len(create_model(fitObj.modelName, **model_kws).stateVars)
    treatmentScheduleList = treatmentScheduleList if treatmentScheduleList is not None else utils.ExtractTreatmentFromDf(
        fitObj.data)
    initialScheduleList = at_kws.get('initialScheduleList', [])
    if 'initialScheduleList' in at_kws.keys(): 
        at_kws.pop('initialScheduleList') # Remove the schedule as it is not an argument to the AT simulation function
        at_kws['t_end'] = at_kws.get('t_end', initialScheduleList[-1][1]+250)
        at_kws['t_span'] = (initialScheduleList[-1][1], at_kws['t_end'])
    n_bootstraps = bootstrapResultsDf.shape[0]

    # 1. Perform bootstrapping
    modelPredictionsMat_mean = np.zeros(
        (n_bootstraps, n_timePoints, n_stateVars+2))  # Array to hold model predictions for CI estimation
    modelPredictionsMat_indv = np.zeros(
        (n_bootstraps, n_timePoints, n_stateVars+2))  # Array to hold model predictions with residual variance for PI estimation
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # Set up the model using the parameters from a bootstrap fit
        tmpModel = create_model(fitObj.modelName, **model_kws)
        currParams = fitObj.params.copy()
        for var in bootstrapResultsDf.columns:
            if var == "SSR": continue
            currParams[var].value = bootstrapResultsDf[var].iloc[bootstrapId]
        tmpModel.SetParams(**currParams)
        # Calculate confidence intervals for model prediction
        if initialConditionsDic is not None: tmpModel.SetParams(**initialConditionsDic)
        if atToProfile is None: # Do prediction on a fixed schedule
            tmpModel.Simulate(treatmentScheduleList=treatmentScheduleList, **kwargs.get('solver_kws', {}))
        else: # Do prediction on an adaptive schedule, which may be different for each replicate, depending on the dynamics
            if len(initialScheduleList)==0: # Simulate AT from t=0
                getattr(tmpModel, 'Simulate_'+atToProfile)(**at_kws, solver_kws=kwargs.get('solver_kws', {}))
            else: # Simulate AT with a prior schedule
                tmpModel.Simulate(treatmentScheduleList=initialScheduleList, **kwargs.get('solver_kws', {}))
                at_kws['refSize'] = at_kws.get('refSize', tmpModel.resultsDf.TumourSize.iloc[-1])
                getattr(tmpModel, 'Simulate_'+atToProfile)(**at_kws, solver_kws=kwargs.get('solver_kws', {}))
        tmpModel.Trim(t_eval=t_eval)
        residual_variance_currEstimate = bootstrapResultsDf['SSR'].iloc[
                                             bootstrapId] / fitObj.nfree  # XXX Not sure this is correct for hierarchical model structure. Thus, PIs not used in paper XXX
        for stateVarId, var in enumerate(['TumourSize']+tmpModel.stateVars):
            modelPredictionsMat_mean[bootstrapId, :, stateVarId] = tmpModel.resultsDf[var].values
            modelPredictionsMat_indv[bootstrapId, :, stateVarId] = tmpModel.resultsDf[var].values + np.random.normal(loc=0,
                                                                                                               scale=np.sqrt(
                                                                                                                   residual_variance_currEstimate),
                                                                                                               size=n_timePoints)
        modelPredictionsMat_mean[bootstrapId, :, -1] = tmpModel.resultsDf['DrugConcentration'].values # Add separately as don't want to add this to the individual prediction matrix

    # 3. Estimate confidence and prediction interval for model prediction
    tmpDicList = []
    # Compute the model prediction for the model with the MLE parameter estimates
    tmpModel.SetParams(**fitObj.params)  # Calculate model prediction for best fit
    if initialConditionsDic is not None: tmpModel.SetParams(**initialConditionsDic)
    if treatmentScheduleList is None: treatmentScheduleList = utils.ExtractTreatmentFromDf(fitObj.data)
    if atToProfile is None: # Do prediction on a fixed schedule
            tmpModel.Simulate(treatmentScheduleList=treatmentScheduleList, **kwargs.get('solver_kws', {}))
    else: # Do prediction on an adaptive schedule, which may be different for each replicate, depending on the dynamics
        if len(initialScheduleList)==0: # Simulate AT from t=0
            getattr(tmpModel, 'Simulate_'+atToProfile)(**at_kws, solver_kws=kwargs.get('solver_kws', {}))
        else: # Simulate AT with a prior schedule
            tmpModel.Simulate(treatmentScheduleList=initialScheduleList, **kwargs.get('solver_kws', {}))
            at_kws['refSize'] = at_kws.get('refSize', tmpModel.resultsDf.TumourSize.iloc[-1])
            getattr(tmpModel, 'Simulate_'+atToProfile)(**at_kws, solver_kws=kwargs.get('solver_kws', {}))
    tmpModel.Trim(t_eval=t_eval)
    for i, t in enumerate(t_eval):
        for stateVarId, var in enumerate(['TumourSize']+tmpModel.stateVars):
            tmpDicList.append({"Time": t, "Variable":var, "Estimate_MLE": tmpModel.resultsDf[var].iloc[i],
                               "DrugConcentration": tmpModel.resultsDf['DrugConcentration'].iloc[i],
                               "CI_Lower_Bound": np.percentile(modelPredictionsMat_mean[:, i, stateVarId], (1 - alpha) * 100 / 2),
                               "CI_Upper_Bound": np.percentile(modelPredictionsMat_mean[:, i, stateVarId],
                                                               (alpha + (1 - alpha) / 2) * 100),
                               "PI_Lower_Bound": np.percentile(modelPredictionsMat_indv[:, i, stateVarId], (1 - alpha) * 100 / 2),
                               "PI_Upper_Bound": np.percentile(modelPredictionsMat_indv[:, i, stateVarId],
                                                               (alpha + (1 - alpha) / 2) * 100)})
            if estimate_fractions and var != "TumourSize":
                # Compute the sensitive/resistant fractions, respectively
                stateVarId_totalSize = 0
                bootstrap_fractions_list = modelPredictionsMat_mean[:, i, stateVarId]/modelPredictionsMat_mean[:, i, stateVarId_totalSize]
                tmpDicList[-1] = {**tmpDicList[-1], 
                                    "Estimate_MLE_Fraction": tmpModel.resultsDf[var].iloc[i]/tmpModel.resultsDf["TumourSize"].iloc[i],
                                    "CI_Lower_Bound_Fraction": np.percentile(bootstrap_fractions_list, (1 - alpha) * 100 / 2),
                                    "CI_Upper_Bound_Fraction": np.percentile(bootstrap_fractions_list, (alpha + (1 - alpha) / 2) * 100),
                                    }
    modelPredictionDf = pd.DataFrame(tmpDicList)
    if returnTrajectories:
        # Format the trajectories for each bootstrap into a data frame
        tmp_list = []
        for bootstrap_id in range(n_bootstraps):
            tmp_df = pd.DataFrame(modelPredictionsMat_mean[bootstrap_id], columns=["TumourSize", "S", "R", "DrugConcentration"])
            tmp_df["Time"] = tmpModel.resultsDf.Time.values
            tmp_df["BootstrapId"] = bootstrap_id
            tmp_list.append(tmp_df)
        trajectories_df = pd.concat(tmp_list)
        return modelPredictionDf, trajectories_df
    else:
        return modelPredictionDf

# ====================================================================================
def benchmark_prediction_accuracy(fitObj, bootstrapResultsDf, dataDf, initialConditionsList=None, model_kws={},
                                  show_progress=True, **kwargs):
    # Initialise
    n_bootstraps = bootstrapResultsDf.shape[0]

    # Compute the r2 value for each bootstrap
    tmpDicList = []
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # Set up the model using the parameters from a bootstrap fit
        tmpModel = create_model(fitObj.modelName, **model_kws)
        currParams = fitObj.params.copy()
        for var in bootstrapResultsDf.columns:
            if var == "SSR": continue
            currParams[var].value = bootstrapResultsDf[var].iloc[bootstrapId]
        if initialConditionsList is not None:
            for var in initialConditionsList.keys():
                currParams[var].value = initialConditionsList[var]

        # Make prediction and compare to true data
        tmpModel.residual = residual(data=dataDf, model=tmpModel, params=currParams,
                                  x=None, feature="Confluence", solver_kws=kwargs.get('solver_kws', {}))
        r2Val = compute_r_sq(fit=tmpModel, dataDf=dataDf, feature="Confluence")

        # Save results
        tmpDicList.append({"Model":fitObj.modelName, "BootstrapId":bootstrapId,
                           "rSquared":r2Val})
    return pd.DataFrame(tmpDicList)

# ====================================================================================
def compute_confidenceInterval_parameters(fitObj, bootstrapResultsDf, paramsToEstimateList=None, alpha=0.95):
    # Initialise
    if paramsToEstimateList is None:
        paramsToEstimateList = [param for param in fitObj.params.keys() if fitObj.params[param].vary]

    # Estimate confidence intervals for parameters from bootstraps
    tmpDicList = []
    for i, param in enumerate(paramsToEstimateList):
        tmpDicList.append({"Parameter": param, "Estimate_MLE": fitObj.params[param].value,
                           "Lower_Bound": np.percentile(bootstrapResultsDf[param].values, (1 - alpha) * 100 / 2),
                           "Upper_Bound": np.percentile(bootstrapResultsDf[param].values,
                                                        (alpha + (1 - alpha) / 2) * 100)})
    return pd.DataFrame(tmpDicList)