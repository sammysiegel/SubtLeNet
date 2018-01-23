import obj 

obj.DEBUG = True 
coll = obj.PFSVCollection()
coll.add_classes(['singletons', 'charged', 'inclusive', 'sv'], '/home/snarayan/hscratch/baconarrays/v13.2/QCD_Pt_80to120_13TeV_pythia8_ext_Output_*_1_XXXX.npy')

coll2 = obj.PFSVCollection()
coll2.add_classes(['singletons', 'charged', 'inclusive', 'sv'], '/home/snarayan/hscratch/baconarrays/v13.2/QCD_Pt_800to1000_13TeV_pythia8_ext_Output_*_1_XXXX.npy')

g = obj.generatePFSV([coll, coll2], batch=2)
for i in xrange(5):
    d = next(g)
    print d[1]
