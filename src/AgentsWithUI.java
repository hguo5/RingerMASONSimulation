import sim.engine.*;
import sim.display.*;
import sim.util.media.chart.*;
import javax.swing.*;
import org.jfree.chart.*;
import org.jfree.data.*;

/**
 *
 * @author Hui
 */
public class AgentsWithUI extends GUIState{

    org.jfree.data.xy.XYSeries series;    // the data series we'll add to
    org.jfree.data.xy.XYSeries series2;    // the data series we'll add to
    TimeSeriesChartGenerator chart;  // the charting facility
    public JFrame chartFrame;
    
    public double total_payoff;
    public double total_happiness;

    public static String chartedParameter = "Average Payoff Per Call";

    public static void main(String[] args){
        AgentsWithUI a = new AgentsWithUI();
        Console c = new Console(a);
        c.setVisible(true);
    }
    
    public AgentsWithUI(){super(new Agents(System.currentTimeMillis()));}
    public AgentsWithUI(SimState state){super(state);}
    public static String getName(){return "Ringer Manager Simulation";}
    
    public void init(Controller c){
        super.init(c);
        chart = new TimeSeriesChartGenerator();
        chart.setTitle(chartedParameter);
        chart.setXAxisLabel("Step");
        chart.setYAxisLabel(chartedParameter);
        chartFrame = chart.createFrame();
        chartFrame.setVisible(true);
        chartFrame.pack();
        c.registerFrame(chartFrame);
    }
    
    public void start(){
        super.start();
        chart.removeAllSeries();
        series = new org.jfree.data.xy.XYSeries(chartedParameter,false,false);
        series2 = new org.jfree.data.xy.XYSeries("Average Happiness Per Call",false,false);
        chart.addSeries(series, null);
        chart.addSeries(series2, null);
        
        //update chart after each step.
        scheduleRepeatingImmediatelyAfter(new Steppable(){
            public void step(SimState state){
                if (state.schedule.getSteps()<=1)
                    return;
                //after each step, chart desired information
                Agents agents = (Agents)state;
                double x = state.schedule.getSteps();
                
                //The parameter that is being monitored. 
                //double y = (double)agents.neighborCount;
                //double y = agents.overallHappiness;
                double y = agents.payoff;
                double z = agents.overallHappiness;
                
                //total_payoff+=y;
                //total_happiness+=z;
                
                if (x >= state.schedule.EPOCH && x < state.schedule.AFTER_SIMULATION){
                   //series.add(x, total_payoff/x, true);
                   //series2.add(x, total_happiness/x, true);
                   series.add(x, y, true);
                   series2.add(x, z, true);
                   chart.updateChartLater(state.schedule.getSteps());
                }
            }
        });
    }
    
    public void finish(){
        super.finish();
        chart.update(state.schedule.getSteps(), true);
        chart.repaint();
        chart.stopMovie();
    }

    public void quit(){
        super.quit();
        chart.update(state.schedule.getSteps(), true);
        chart.repaint();
        chart.stopMovie();
        if (chartFrame != null)	chartFrame.dispose();
        chartFrame = null;
    }
}
