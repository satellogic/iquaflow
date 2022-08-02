from typing import Optional

import pandas as pd
import seaborn as sns


class ExperimentVisual:
    """
    This objects allows the user to manage the experiment visualization.
    
    Args:
        df: pd.DataFrame. Dataframe containing the data to visualize.
        The user should add a dataframe that is suitable for the intended visualization.
        This is x and y are indexes and content in the dataframe.
        The legend is either a column name or an aggregated dataframe with mean and std for the variable of interest.
        out_fullfn: str. Output fullfilename.

    Attributes:
        df: pd.DataFrame. Dataframe containing the data to visualize
        out_fullfn: str. Output fullfilename.
    """

    def __init__(self, df: pd.DataFrame, out_fullfn: Optional[str] = None):
        self.df = df
        self.out_fullfn = out_fullfn

    def _adapt_agg(
        self, var: Optional[str] = "val_rmse", only_mean: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        This is an internal method that adapts an aggregated Multiindex dataframe for some specific visualizations.
        
        Args:
            var (Optional[str]): Variable label that will contain the yaxis values in a plot.
            only_mean (Optional[bool]): Whether if the adapter considers only mean or also +/- std.
        
        Returns:
            Adapted dataframe.
        """
        df1 = self.df
        a, b, c = [], [], []
        for i, row in df1.iterrows():
            m = row[var]["mean"]
            s = row[var]["std"]
            a.append(m - s)
            b.append(m)
            c.append(m + s)

        values = [list(a) for a in df1.index.tolist()]
        columns = [n for n in df1.index.names]
        df0 = pd.DataFrame(values, columns=columns)
        if only_mean:
            df2 = df0
            df2[var] = b
        else:
            df2 = pd.concat([df0, df0, df0])
            df2[var] = a + b + c
        return df2

    def visualize(
        self,
        plot_kind: Optional[str] = "lineplot",
        xvar: Optional[str] = "ds_modifier",
        yvar: Optional[str] = "val_rmse",
        legend_var: Optional[str] = "min_size",
        title: Optional[str] = None,
        plot_mean_std: Optional[bool] = False,
    ) -> None:
        """
        Args:
            plot_kind (Optional[str]): Either 'lineplot', 'bars' or 'scatter'. Also default types from pandas plot method when plot_mean_std is True.
            xvar (Optional[str]): Label variable corresponding to the x axis, it is an index in the dataframe.
            yvar (Optional[str]): Label variable corresponding to the y axis. This is the content values in the dataframe.
            legend_var (Optional[str]): The column names in the dataframe. This parameter is used when plot_mean_std is set to False.
            title (Optional[str]): Optional title, default is None.
            plot_mean_std (Optional[bool]): Whether if the user wants to plot mean and std from an aggregated dataframe.
        """
        df = self.df
        is_agg = df.columns.nlevels > 1

        if plot_mean_std:

            # Plot mean_std
            if not is_agg:
                raise ValueError(
                    "ERROR > plot_mean_std requires aggregated dataframe with std and mean"
                )
            elif plot_kind in ["lineplot", "line"]:
                handle = df[yvar].plot(kind="line")
            elif plot_kind in ["bars", "bar"]:
                handle = df[yvar].plot(kind="bar")
            else:
                try:
                    handle = df[yvar].plot(kind=plot_kind)
                except Exception as e:
                    print(e)
            _ = handle.set_ylabel(yvar)

        else:

            # Don't plot mean_std but other stuff

            if plot_kind == "lineplot":
                if is_agg:
                    # Aggregated case
                    df = self._adapt_agg(var=yvar)
                    handle = sns.lineplot(
                        data=df, x=xvar, y=yvar, hue=legend_var, ci=100
                    )
                else:
                    # Normal case
                    handle = sns.lineplot(data=df, x=xvar, y=yvar, hue=legend_var)
            elif plot_kind == "bars":
                if not is_agg:
                    # Normal case, must agg
                    df = df.groupby([xvar, legend_var]).agg({yvar: ["mean", "std"]})
                # Aggregated case
                df = df.unstack().drop(columns="std", level=1)
                df = df.droplevel(level=0, axis=1)
                df = df.droplevel(level=0, axis=1)
                handle = df.plot(kind="bar")
                _ = handle.set_ylabel(yvar)
            elif plot_kind == "scatter":
                if is_agg:
                    df = self._adapt_agg(var=yvar, only_mean=True)
                handle = sns.scatterplot(data=df, x=xvar, y=yvar, hue=legend_var)

        if title:
            _ = handle.set_title(title)

        if self.out_fullfn:
            figure = handle.get_figure()
            figure.savefig(self.out_fullfn)
