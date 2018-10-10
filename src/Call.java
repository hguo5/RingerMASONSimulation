import sim.util.*;

/**
 *
 * @author Hui
 */
public class Call {
    public Agent caller;
    public Agent callee;
    public boolean urgency;
    
    public int action; //0 for ignored, 1 for answered.
    public Bag feedbacks;
    
    public int location; //keep location, since agents move around
    public long step; //keep step number
    
    public Call(Agent caller, Agent callee, boolean urgency, long step){
        this.caller = caller;
        this.callee = callee;
        this.urgency = urgency;
        this.action = -1;
        this.feedbacks = new Bag();
        this.location = callee.location;
        this.step = step;
    }
    
    public Call(Agent caller, Agent callee, boolean urgency, int location, long step){
        this.caller = caller;
        this.callee = callee;
        this.urgency = urgency;
        this.action = -1;
        this.feedbacks = new Bag();
        this.location = location;
        this.step = step;
    }
    
    public boolean isFamily(){
        return caller.familyCircle==callee.familyCircle;
    }
    
    public boolean isColleague(){
        return caller.colleagueCircle==callee.colleagueCircle;
    }
    
    public boolean isFriend(){
        return caller.friendCircle==callee.friendCircle;
    }
    
    public boolean isStranger(){
        if (isFamily()||isColleague()||isFriend())
            return false;
        else
            return true;
    }
}
