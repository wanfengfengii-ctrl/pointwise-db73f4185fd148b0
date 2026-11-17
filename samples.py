def E(v):
    return {"kind":"exact","value":v}

def L(v):
    return {"kind":"left","upper":v}

def R(v):
    return {"kind":"right","lower":v}

def I(a,b):
    return {"kind":"interval","lower":a,"upper":b}

def exact_rows(cohort="exact"):
    points = [(1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]
    return [{"cohort_id":cohort,"sample_id":str(j),"rev":1,"x":[E(a),E(b)],"count":1}
            for j,(a,b) in enumerate(points)]

def mixed_rows(cohort="mixed"):
    triples = [(E(1),E(1),4),(E(-1),E(-1),4),(E(1),E(0),4),
               (E(-1),E(0),4),(E(0),E(1),4),(E(0),E(-1),4),
               (E(-.4),L(.2),2),(E(.8),R(.5),3),(I(-.8,.1),E(-.3),2),
               (L(-.2),I(-1,.4),1),(R(.1),R(.2),2),(I(-.4,.6),L(.3),2)]
    return [{"cohort_id":cohort,"sample_id":str(j),"rev":1,"x":[a,b],"count":n}
            for j,(a,b,n) in enumerate(triples)]

REFERENCE = {"mu":[.054220399820,.062317514860],
             "covariance":[[.581417380321,.354136353153],[.354136353153,.655918716013]],
             "log_likelihood":-72.274849256408,"thresholds":[.15,.25],"probability":.278167435769}
