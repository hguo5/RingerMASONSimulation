import sim.engine.*;
import sim.util.*;

import java.util.*;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

/**
 *
 * @author Hui
 */
public class Agent implements Steppable{
    
    public int remainingSteps = 0;
    public double callRate = 0.05;
    public int location = -1;
    
    public int familyCircle = -1;
    public int colleagueCircle = -1;
    public int friendCircle = -1;
    
    public Bag myFamilies = new Bag();
    public Bag myColleagues = new Bag();
    public Bag myFriends = new Bag();
    public Bag myStrangers = new Bag();
    
    //neighboring calls to which this agent needs to give feedbacks
    public Bag neighboringCalls = new Bag();
    
    //history of calls this agent receives (as a callee)
    //not used for now
    public Bag callHistory = new Bag();
    
    //history of records for classification
    //public Bag records = new Bag();
    
    //Weka dataset
    public Instances data;

    //current neighbors;
    public Bag currentNeighbors = new Bag();
    
    //lock to avoid being called twice in a step. 
    public boolean isCalled = false;
    public Call currentCall = null;
    
    public int id = -1;
    
    public Agent(){
        location = -1;
        remainingSteps = 0;
        this.id = -1;
        initDataset();
    }
    
    public Agent(int id){
        location = -1;
        remainingSteps = 0;
        this.id = id;
        initDataset();
    }
    
    //Initialize the Weka dataset
    public void initDataset(){
        //Attribute list
        ArrayList<Attribute> alist = new ArrayList<Attribute>();
        Attribute aa;
        
        //location
        aa = new Attribute("location", new ArrayList<String>(
                Arrays.asList(Agents.locations)));
        alist.add(aa);
        
        //caller relation
        aa = new Attribute("caller_relation", new ArrayList<String>(
                Arrays.asList(RecordForLearning.relationTypes)));
        alist.add(aa);
        
        //urgency
        List<String> tf = new ArrayList<String>();//True of False attributes
        tf.add("true");tf.add("false");
        aa = new Attribute("urgency", tf);
        alist.add(aa);
        //exists_family
        aa = new Attribute("exists_family", tf);
        alist.add(aa);
        //exists_colleague
        aa = new Attribute("exists_colleague", tf);
        alist.add(aa);
        //exists_friend
        aa = new Attribute("exists_friend", tf);
        alist.add(aa);
        //answer or not
        aa = new Attribute("answer", tf);
        alist.add(aa);
        //payoff, numeric
        aa = new Attribute("@Class@");
        alist.add(aa);
        
        this.data = new Instances("Call Record", alist, 0);
        data.setClassIndex(data.numAttributes()-1);
    }
    
    //Add a record to Weka dataset
    public void addRecord(RecordForLearning rec){
        double[] one = new double[data.numAttributes()];
        
        //location
        one[0] = rec.location;
        //caller relation
        one[1] = rec.callerRelation;
        //urgency
        one[2] = rec.urgency?0:1;//note that 0 is for true
        //exists family
        one[3] = rec.existsFamily?0:1;
        //exists colleague
        one[4] = rec.existsColleague?0:1;
        //exists friend
        one[5] = rec.existsFriend?0:1;
        //answer or not?
        one[6] = 1-rec.action;
        //payoff
        one[7] = rec.getPayoff();
        
        data.add(new DenseInstance(1.0, one));
    }
    
    public void step(SimState state){
        
        Agents agents = (Agents)state;
        double x = 0.0;
        
        //Enter a random place
        if (remainingSteps<=0){
            int sum = 0;
            int i;
            for(i=0;i<agents.locationWeights.length; i++){
                sum+=agents.locationWeights[i];
            }
            x = agents.random.nextDouble();
            int y = 0;
            for(i=0;i<agents.locationWeights.length;i++){
                y += agents.locationWeights[i];
                if (x<=(double)y/(double)sum)
                    break;
            }
            if (i>=agents.locationWeights.length)
                i=agents.locationWeights.length;
            
            //location = loction type id * number of agents + location id
            //e.g., meeting #1 with 1000 agents = 1*1000+1=1001
            //75% probability, agent enters own home/meeting/party
            //25% probability, agent enters another random home/meeting/party
            x = agents.random.nextDouble();
            switch(i){
                case 0: //home
                    x = x*agents.numHomes*4;
                    if (x>=agents.numHomes)
                        location = i*agents.numAgents+familyCircle;
                    else
                        location = i*agents.numAgents+(int)x;
                    break;
                case 1: //meeting
                    x = x*agents.numMeetings*4;
                    if (x>=agents.numMeetings)
                        location = i*agents.numAgents+colleagueCircle;
                    else
                        location = i*agents.numAgents+(int)x;
                    break;
                case 2: //party
                    x = x*agents.numParties*4;
                    if (x>=agents.numParties)
                        location = i*agents.numAgents+friendCircle;
                    else
                        location = i*agents.numAgents+(int)x;
                    break;
                default:
                    location = i*agents.numAgents;
            }
            remainingSteps = (int)(agents.random.nextGaussian()*30+60.5);
            if (remainingSteps>90) remainingSteps = 90;
            if (remainingSteps<30) remainingSteps = 30;
            remainingSteps *= agents.locationWeights[i];
            
        }else{
            remainingSteps --;
        }
        
        //Output one agent's info
        /*
        if (this.id==0){
            System.out.println("Location: "+agents.locations[location/agents.numAgents]
                    +" #"+(location%agents.numAgents));
            System.out.println("Remaining: "+remainingSteps);
        }*/
        
        //Once every agent enters a place...
        if (state.schedule.getSteps()<=0) return;
        
        //As a caller
        //Randomly make a random call
        x = agents.random.nextDouble();
        if (x<=callRate){
            
            //25% agent calls family, 25% colleague, 25% friend, 25% stranger
            x = agents.random.nextDouble();
            Bag temp;
            if (x<0.25)
                temp = this.myFamilies;
            else if (x<0.5)
                temp = this.myColleagues;
            else if (x<0.75)
                temp = this.myFriends;
            else
                temp = this.myStrangers;
            
            x = agents.random.nextDouble();
            Agent callee = (Agent)temp.get((int)(x*temp.size()));
            //Caller and callee should not be in the same place
            while(callee.location==this.location){
                temp = this.myStrangers;//to avoid all group members being in the same place
                x = agents.random.nextDouble();
                callee = (Agent)temp.get((int)(x*temp.size()));
            }
            
            /*
            Agent callee= (Agent)agents.allAgents.get((int)(x*agents.numAgents));
            //Caller and callee should not be in the same place
            while(callee.location==this.location){
                x = agents.random.nextDouble();
                callee= (Agent)agents.allAgents.get((int)(x*agents.numAgents));
            }*/
            
            //make sure that each agent is only called once in each step
            if (!callee.isCalled){
                x = agents.random.nextDouble();
                Call call = new Call(this, callee, x<0.5, state.schedule.getSteps());

                //The callee makes a decision (whether or not to take this call).
                callee.handleACall(call, state);
                agents.callsInThisStep.add(call);

                //Keep history. Disabled for now to save space
                //callee.callHistory.add(call);
                
                callee.isCalled = true;
                callee.currentCall = call;

                //Add this call to all of the callee's neighbors
                callee.currentNeighbors = agents.getNeighbors(callee.location);
                Agent neighbor;
                for(int i=0;i<callee.currentNeighbors.size();i++){
                    neighbor = (Agent)callee.currentNeighbors.get(i);
                    if (neighbor.id!=callee.id)
                        neighbor.neighboringCalls.add(call);
                }
            }
        }
        
        //As a neigbhor
        //Respond to neighbor calls with feedbacks. 
        //Move this step to after all agents have made a call. 
        //giveFeedbacks(state);
    }
    
    public void handleACall(Call call, SimState state){

        Agents agents = (Agents)state;
        
        //Agent will make a decision based on call info, 
        //as well as call history
        //do adaptive learning based on feedbacks.
        
        //decision being whether or not to answer the call
        //0 for ignored, 1 for answered.
        
        //One method is that the action is always random.
        //call.action = state.random.nextBoolean()?1:0;

        /*
        To begin with, let's assume the agents comply with the following norms:
        -- Answer calls if the agent is at home, parties or diner, and
        -- Ignore calls otherwise(meeting or library)
        -- Ignore calls if casual from strangers, answer otherwise.
        */
        boolean basedonloc = true;
        boolean basedoncall = true;
        
        //Originally, the fixed norms are:
        //If in a meeting or a library, not answer
        //Otherwise, answer
        if ((int)(location/Agents.numAgents)==1||(int)(location/Agents.numAgents)==3){
            basedonloc = false;
        }
        
        //Later, we decided to use the following norms:
        //  -- If in a meeting or a library, definitely ignore;
        //  -- If in an ER, definitely answer;
        //  -- If at home, more likely to answer(67% answer vs 33% ignore);
        //  -- If at a party, more likely to ignore(33% answer vs 67% ignore)
        
        double x = state.random.nextDouble();
        switch((int)(location/Agents.numAgents)){
            //at home
            case 0:
                basedonloc = x<0.67;
                break;
            //in a meeting
            case 1:
                basedonloc = false;
                break;
            //at a party
            case 2:
                basedonloc = x>0.67;
                break;
            //in a library
            case 3:
                basedonloc = true;
                break;
            //in an ER
            case 4:
                basedonloc = false;
                break;
            default: break;
        }
        
        //basedoncall is based on Callee Payoff, 
        // (choose the action with higher payoff)
        //which is the same as: 
        //if (call.isStranger()&&(!call.urgency))
        //    basedoncall = false;
        if (call.isStranger()){
            if (call.urgency)
                basedoncall = agents.payoff_a[6]>agents.payoff_a[7];
            else
                basedoncall = agents.payoff_a[4]>agents.payoff_a[5];
        }
        else{
            if (call.urgency)
                basedoncall = agents.payoff_a[2]>agents.payoff_a[3];
            else
                basedoncall = agents.payoff_a[0]>agents.payoff_a[1];
        }

        if (basedonloc==basedoncall)
            call.action = basedonloc?1:0;
        else
            call.action = state.random.nextBoolean()?1:0;
        
        //A better way to make a decision, with adaptive learning, 
        //is using Weka's classification methods.
        //Learning starts after a learning period
        if ((Agents.simulationNumber>=2)&&(this.data.numInstances()>Agents.learningPeriod)){
            int temp = getAction(call,state);
            if (temp>=0)
                call.action = temp;
        }
    }
    
    public void giveFeedbacks(SimState state){
        Agents agents = (Agents)state;
        Bag todo = new Bag(neighboringCalls);
        neighboringCalls = new Bag();
        if (todo.size()<=0)
            return;
        Call call;
        Feedback temp;
        
        //boolean feedback = true;
        //UPDATE: Now we use payoff instead of boolean feedback;
        double payoff = 0.0;
        
        for(int i=0;i<todo.size();i++){
            call = (Call)todo.get(i);
            
            //decide a feedback based on call info
            //One solution is that the feedback is always random:
            //feedback = state.random.nextBoolean();
            
            //To begin with, let's assume that agents give feedbacks
            //solely based on locations:
            //positive if answered at home, parties, diner; random if ignored
            //negative if answered at meeting or library; random if ignored
            
            /*
            if ((int)(location/Agents.numAgents)==1||(int)(location/Agents.numAgents)==3){
                if (call.action==1)
                    feedback = false;
                else
                    feedback = state.random.nextBoolean();
            }
            else{
                if (call.action==1)
                    feedback = true;
                else
                    feedback = state.random.nextBoolean();
            }*/
            
            
            //UPDATED: now we use payoffs as feedbacks
            //In the 2nd simulation: 
            //remeber that, in meeting, library and party, 
            //people think callee should ignore
            int l = (int)(call.location/Agents.numAgents);
            if (call.action==1){
                switch(l){
                    case 1: 
                        payoff = agents.payoff_i[12+2*l];
                        break;
                    case 2:
                        payoff = agents.payoff_i[12+2*l];
                        break;
                    case 3:
                        payoff = agents.payoff_i[12+2*l];
                        break;
                    default:
                        payoff = agents.payoff_a[12+2*l];
                        break;
                }
            }
            else{
                switch(l){
                    case 1: 
                        payoff = agents.payoff_i[13+2*l];
                        break;
                    case 2: 
                        payoff = agents.payoff_i[13+2*l];
                        break;
                    case 3:
                        payoff = agents.payoff_i[13+2*l];
                        break;
                    default:
                        payoff = agents.payoff_a[13+2*l];
                        break;
                }
            }
            
            //In the 3rd simulation, 
            //neighbor hears the explanation
            if ((Agents.simulationNumber==3)&&(this.data.numInstances()>Agents.learningPeriod)){
                //get the action that the neighbor would take
                int action = getAction(call, state);
                //If neighbor thinks callee should answer
                if (action==1){
                    //callee answers or not
                    payoff = (call.action==1?agents.payoff_a[12+2*l]:agents.payoff_a[13+2*l]);
                }else if (action == 0){
                    payoff = (call.action==1?agents.payoff_i[12+2*l]:agents.payoff_i[13+2*l]);
                }
                //action could be -1, in which case no action is taken. 
            }
            
            //temp = new Feedback(call, this, feedback);
            temp = new Feedback(call, this, payoff);
            call.feedbacks.add(temp);
        }
    }
    
    public String getLocationString(){
        return Agents.locations[location/Agents.numAgents]
                +" #"+(location%Agents.numAgents);
    }
    
    //Decided, based on history, which action is better
    public int getAction(Call call, SimState state){
        Agents agents = (Agents)state;
        int action = -1;
        Classifier cls = new LinearRegression();
        RecordForLearning rec = new RecordForLearning(call, agents);
        double[] one = new double[data.numAttributes()];
        //location
        one[0] = rec.location;
        //caller relation
        one[1] = rec.callerRelation;
        //urgency
        one[2] = rec.urgency?0:1;//note that 0 is for true
        //exists family
        one[3] = rec.existsFamily?0:1;
        //exists colleague
        one[4] = rec.existsColleague?0:1;
        //exists friend
        one[5] = rec.existsFriend?0:1;
        //What if I ignore?
        one[6] = 1;
        //payoff
        one[7] = 0;//0 for now

        try{
            cls.buildClassifier(data);
            //What if I ignore?
            one[6] = 1;
            double a1 = cls.classifyInstance(new DenseInstance(1.0, one));
            //What if I answer?
            one[6] = 0;
            double a2 = cls.classifyInstance(new DenseInstance(1.0, one));

            //choose the action with higher predicted overall payoff
            if (a1>a2)
                action = 0;
            else
                action = 1;
        }
        catch(Exception e){
            //do nothing
        }
        return action;
    }
}
