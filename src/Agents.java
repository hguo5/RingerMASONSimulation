import sim.engine.*;
import sim.util.*;
import sim.field.continuous.*;
import sim.field.network.*;

import java.io.*;
import java.util.*;

/**
 *
 * @author Hui
 */
public class Agents extends SimState {
    
    //Which simulation are we doing?
    public static int simulationNumber = 1;
    
    //Main parameters
    //Changed from 1000,20,20,20
    public static int numAgents = 1000;
    public static int numHomes = 20;
    public static int numMeetings = 20;
    public static int numParties = 20;
    
    public static double callRate = 0.05;
    
    //Location names
    public static String[] locations = {"home", "meeting","party","library","ER"};
    
    //location weights are multipliers for probabilities and durations.
    //Use uniform weights for now. Originally {8,2,3,2,1}.
    public static int[] locationWeights = {1,1,1,1,1};
    
    //Agent will start to learn (instead of using fixed norms) 
    //after the learning period.
    public static int learningPeriod = 50;
    
    //Average the output in a window (#steps)
    public static int windowSize = 200;
    public static int windowSize2 = 200;
    
    public LinkedHashMap<Long, Double> window = new LinkedHashMap<Long, Double>(){
        @Override
        protected boolean removeEldestEntry(Map.Entry<Long, Double> eldest){
            return this.size() > windowSize;
        }
    };
    
    public LinkedHashMap<Long, Integer> numCalls = new LinkedHashMap<Long, Integer>(){
        @Override
        protected boolean removeEldestEntry(Map.Entry<Long, Integer> eldest){
            return this.size() > windowSize;
        }
    };
    
    public LinkedHashMap<Long, Double> window2 = new LinkedHashMap<Long, Double>(){
        @Override
        protected boolean removeEldestEntry(Map.Entry<Long, Double> eldest){
            return this.size() > windowSize2;
        }
    };
    
    //cost of giving feedbacks
    public static double costFeedback = 0.5; //not used for now
    
    //cost of giving feedbacks and giving explanation;
    public static double costExplanation = 0.6; //not used for now
    
    //pay off table, if neighbor thinks callee should answer
    public double[] payoff_a = new double[22];
    //pay off table, if neighbor thinks callee should ignore
    //during first and second simulations, neighbor thinks callee should ignore
    //during a meeting or in a library (hard-coded in this way).
    public double[] payoff_i = new double[22];
    
    //reference to all agents in the simulation
    public Bag allAgents = new Bag();
    
    //keep all calls happening in current step
    public Bag callsInThisStep = new Bag();
    
    //sum of #neighbors of all calls in a step
    //different from #neighbors involved in calls
    public int neighborCount = 0;
    //average happiness of involved people in a step
    public double overallHappiness = 0.0;
    //total payoff in a step
    public double payoff = 0.0;
    
    //Write results to file
    public BufferedWriter out = null;
    
    public Agents(long seed){
        super(seed);
    }
    
    //executed before steppings. 
    public void start(){
        super.start();
        
        //read in the payoff table. 
        readPayoff();
        
        try{
            out = new BufferedWriter(new FileWriter("Results_Sim"+simulationNumber+".csv"));
            out.write("Step,#Calls,Payoff Per Call,Happiness Per Call,Avg Payoff in Window,Avg Happiness in Window\r\n");
        }catch(Exception e){
            try{
                out.close();
            }catch(Exception e2){
            }
            out = null;
        }
        
        //reset everything
        window.clear();
        window2.clear();
        allAgents = new Bag();
        callsInThisStep = new Bag();
        neighborCount = 0;
        overallHappiness = 0.0;
        payoff = 0.0;
        
        //initialize all agents
        for(int i=0; i<numAgents; i++){
            Agent agent = new Agent(i);
            
            //define networks
            agent.familyCircle = (int)(i/numHomes);
            agent.colleagueCircle = i % numMeetings;
            //friend circle is random. To be updated.
            agent.friendCircle = (int)(random.nextDouble()*numParties);
            
            //Randomize call rate
            agent.callRate = random.nextGaussian()*callRate/5+callRate;
            allAgents.add(agent);
        }
        
        //Each agent keeps lists of their family, colleagues and friends
        Agent temp;
        int i = 0;
        for(i=0; i<numAgents; i++){
            Agent agent = (Agent)allAgents.get(i);
            for(int j=0; j<allAgents.size();j++){
                if (i==j) continue;
                temp = (Agent)allAgents.get(j);
                
                //keep references to members in my circles. 
                if (agent.familyCircle==temp.familyCircle)
                    agent.myFamilies.add(temp);
                else if (agent.colleagueCircle==temp.colleagueCircle)
                    agent.myColleagues.add(temp);
                else if (agent.friendCircle==temp.friendCircle)
                    agent.myFriends.add(temp);
                else
                    agent.myStrangers.add(temp);
            }
            schedule.scheduleRepeating(agent, 0, 1.0);
        }
        
        //give feedback after all calls are made
        schedule.scheduleRepeating(new Steppable(){
            public void step(SimState state){
                Agents agents = (Agents)state;
                Agent temp;
                for(int i=0;i<agents.allAgents.size();i++){
                    temp = (Agent)agents.allAgents.get(i);
                    temp.giveFeedbacks(state);
                }
            }
        }, 1, 1.0);
        
        //after each step, output information and 
        //reset isCalled and currentCall of each agent.
        schedule.scheduleRepeating(new Steppable(){
            public void step(SimState state){
                if (state.schedule.getSteps()<=1)
                    return;
                Agents agents = (Agents)state;
                Agent temp;
                
                //Sum up the number of neighbors during each call. 
                //One agent may be counted multiple times. 
                neighborCount = 0;
                RecordForLearning record;
                for(int i=0;i<agents.allAgents.size();i++){
                    temp = (Agent)agents.allAgents.get(i);
                    if (temp.isCalled){
                        
                        //if called, agent keeps a record of this call for 
                        //future classification
                        record = new RecordForLearning(temp.currentCall, agents);
                        if (temp.id==0)
                            System.out.println("Record ID: "+temp.data.numInstances()+"\t"+record.toCSVString());

                        //temp.records.add(record);
                        //also add to Weka dataset
                        temp.addRecord(record);
                        
                        neighborCount += temp.currentNeighbors.size();
                        temp.isCalled = false;
                        temp.currentCall = null;
                    }
                }
                
                //Calculate the overall happiness in this step. 
                //Caller happiness: +1 if answered, -1 if ignored.
                //Neighbors: based on feedbacks. 
                //Say there are N neighbors. For each neighbor, 
                //if positive feedback, +1/N, otherwise, -1/N
                overallHappiness = 0.0;
                int peopleinvolved = 0;
                Call call;
                Feedback feedback;
                for(int i=0;i<agents.callsInThisStep.size();i++){
                    call = (Call)agents.callsInThisStep.get(i);
                    if (call.action==1){
                        overallHappiness+=1.0;
                    }
                    else{
                        overallHappiness-=1.0;
                    }
                    peopleinvolved++;
                    
                    for(int j=0;j<call.feedbacks.size();j++){
                        feedback = (Feedback)call.feedbacks.get(j);
                        if (feedback.payoff>0){
                            overallHappiness += 1.0/call.feedbacks.size();
                        }
                        else if (feedback.payoff<0){
                            overallHappiness -= 1.0/call.feedbacks.size();
                        }
                        peopleinvolved++;
                    }
                }
                
                //get average happiness per call
                //if (peopleinvolved>0)
                //    overallHappiness/=(double)peopleinvolved;
                if (agents.callsInThisStep.size()>0)
                	overallHappiness/=(double)agents.callsInThisStep.size();

                //Calculate payoff
                payoff = 0.0;
                int multiplier = 1;
                for(int i=0;i<agents.callsInThisStep.size();i++){
                    call = (Call)agents.callsInThisStep.get(i);
                    //Answer call:
                    if (call.action==1){
                        //Callee payoff
                        if (call.isStranger())
                            payoff+=(call.urgency?payoff_a[6]:payoff_a[4]);
                        else
                            payoff+=(call.urgency?payoff_a[2]:payoff_a[0]);
                        //Caller payoff
                        payoff+=(call.urgency?payoff_a[10]:payoff_a[8]);
                        
                        //In first and second simulations, 
                        //Neighbor thinks callee should ignore calls during a meeting, 
                        //in a library, or at a party.
                        /*
                        int l = (int)(call.location/Agents.numAgents);
                        switch(l){
                            case 1: 
                                payoff += multiplier*payoff_i[12+2*l];
                                break;
                            case 2: 
                                payoff += multiplier*payoff_i[12+2*l];
                                break;
                            case 3:
                                payoff += multiplier*payoff_i[12+2*l];
                                break;
                            default:
                                payoff += multiplier*payoff_a[12+2*l];
                                break;
                        }*/
                    }
                    //Ignore call:
                    else{
                        //Callee payoff
                        if (call.isStranger())
                            payoff+=(call.urgency?payoff_a[7]:payoff_a[5]);
                        else
                            payoff+=(call.urgency?payoff_a[3]:payoff_a[1]);
                        //Caller payoff
                        payoff+=(call.urgency?payoff_a[11]:payoff_a[9]);
                        
                        //In first and second simulations, 
                        //Neighbor thinks callee should ignore calls during a meeting, 
                        //in a library, or at a party.
                        /*
                        int l = (int)(call.location/Agents.numAgents);
                        switch(l){
                            case 1: 
                                payoff += multiplier*payoff_i[13+2*l];
                                break;
                            case 2: 
                                payoff += multiplier*payoff_i[13+2*l];
                                break;
                            case 3:
                                payoff += multiplier*payoff_i[13+2*l];
                                break;
                            default:
                                payoff += multiplier*payoff_a[13+2*l];
                                break;
                        }*/
                    }

                    //UPDATE: Use the actual payoffs the neighbors gave
                    multiplier = call.feedbacks.size();
                    double n = 0.0;
                    for(int j=0;j<call.feedbacks.size();j++){
                        n += ((Feedback)call.feedbacks.get(j)).payoff;
                    }
                    if (multiplier>0)
                    	n /= multiplier;
                    payoff += n;
                }
                
                //When calculating overallHappiness, we don't consider callees.
                //When calculating payoffs, callees are also involved. 
                //peopleinvolved += agents.callsInThisStep.size();
                //get average payoff of all involved people
                //if (peopleinvolved>0)
                //    payoff/=(double)peopleinvolved;
                //UPDATE: average payoff per call
                if (agents.callsInThisStep.size()>0)
                	payoff /= (double)agents.callsInThisStep.size();
                
                
                //System.out.println("Step: "+state.schedule.getSteps()+"\t#calls: "+agents.callsInThisStep.size());
                //System.out.println("Step: "+state.schedule.getSteps()+"\t#neighbors: "+neighborCount);
                //System.out.println();
                
                if (out!=null){
                    try{
                        out.write(""+state.schedule.getSteps()+","+agents.callsInThisStep.size()+","+payoff+","+overallHappiness);
                    }catch(Exception e){
                        try{out.close();}catch(Exception e2){}
                        out = null;
                    }
                }
                //System.out.print(""+state.schedule.getSteps()+","+payoff+","+overallHappiness);
                
                //Output the average payoff over a window
                window.put(state.schedule.getSteps(), payoff);
                numCalls.put(state.schedule.getSteps(), agents.callsInThisStep.size());
                
                //Use mean of window, considering #calls in each step
                payoff = 0.0;
                int callcount = 0;
                int totalcallcount = 0;
                for(Map.Entry<Long, Double> entry : window.entrySet()){
                	callcount = numCalls.get(entry.getKey());
                    payoff+=entry.getValue()*callcount;
                    totalcallcount+=callcount;
                }
                if (totalcallcount>0)
                //payoff /= window.size();
                	payoff /= totalcallcount;
                
                //Use average over window2 as output
                window2.put(state.schedule.getSteps(), overallHappiness);
                //Use mean of window, considering #calls in each step
                overallHappiness = 0.0;
                for(Map.Entry<Long, Double> entry : window2.entrySet()){
                	callcount = numCalls.get(entry.getKey());
                    overallHappiness+=entry.getValue()*callcount;
                }
                //overallHappiness /= window2.size();
                if (totalcallcount>0)
                	overallHappiness /= totalcallcount;
                
                //Use median of window
                /*
                Double x[] = window.values().toArray(new Double[window.size()]);
                Arrays.sort(x);
                payoff = x[(int)(x.length/2)];*/
                
                //System.out.println("Step: "+state.schedule.getSteps()+"\tOverall Happiness: "+overallHappiness);
                //System.out.println("Step: "+state.schedule.getSteps()+"\tPayoff: "+payoff);
                //System.out.println(","+payoff+","+overallHappiness);
                if (out!=null){
                    try{
                        out.write(","+payoff+","+overallHappiness+"\r\n");
                    }catch(Exception e){
                        try{out.close();}catch(Exception e2){}
                        out = null;
                    }
                }
                
                callsInThisStep = new Bag();
            }
        }, 2, 1.0);
        System.out.println("Simulation started.");
    }
    
    //Get all agents currently in a location
    public Bag getNeighbors(int location){
        Bag neighbors = new Bag();
        Agent temp;
        for(int i=0;i<allAgents.size();i++){
            temp = (Agent)allAgents.get(i);
            if (temp.location==location)
                neighbors.add(temp);
        }
        return neighbors;
    }
    
    public void readPayoff(){
        payoff_a = new double[]{1,0,2,-1,0,0.5,1,-0.5,1,-1,2,-2,0,0,1,-1,0,0,1,-1,0,0};
        payoff_i = new double[]{1,0,2,-1,0,0.5,1,-0.5,1,-1,2,-2,0,0,-1,1,0,0,-1,1,0,0};
        try{
            BufferedReader reader = new BufferedReader(new FileReader("payoff.txt"));
            String line;
            String[] items;
            int i = 0;
            while((line=reader.readLine())!=null){
                if (i>=payoff_a.length) break;
                line = line.trim();
                if (line.length()<=0) continue;
                if (line.startsWith("#")) continue;
                items = line.split("\\s+");
                payoff_a[i] = Double.parseDouble(items[0]);
                try{
                    payoff_i[i] = Double.parseDouble(items[1]);
                }catch(Exception ee){
                    payoff_i[i] = payoff_a[i];
                }
                i++;
            }
            reader.close();
        }catch(Exception e){
            e.printStackTrace();
        }
    }
    
    public void finish(){
        super.finish();
        try{out.close();}catch(Exception e){}
        out = null;
    }
    
    public static void main(String[] args){
        doLoop(Agents.class, args);
        System.exit(0);
    }
}