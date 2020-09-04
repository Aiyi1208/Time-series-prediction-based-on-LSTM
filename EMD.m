Nstd = 0.5;
NE = 200; % # of ensemble
numImf =6; % # of imfs
runCEEMD = 0;
maxSift = 10;
typeSpline = 2;
toModifyBC = 2;
randType = 2;
seedNo = 0;
checksignal = 1;
t = 1:length(X1);

[imf] = rcada_eemd(X1,Nstd,NE,numImf); % run EEMD [3]
% [imf] = rcada_eemd(x,0,1,numImf); % run EMD [2]
% [imf] = rcada_eemd(x,Nstd,NE,numImf,runCEEMD,maxSift,typeSpline,toModifyBC,randType,seedNo,checksignal); % EEMD [3]
% [imf] = rcada_eemd(x,Nstd,NE,numImf,1,maxSift,typeSpline,toModifyBC,randType,seedNo,checksignal); % CEEMD [4]
X33=sum(imf(1:numImf,:))'-imf(1,:)';%-imf(1,:)';
X34=X33-imf(2,:)';
