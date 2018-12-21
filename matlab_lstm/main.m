X0 = [-pi/2+.1;0;0;0];
[t,y] = ode45(@(t,y)acrobot_lstm(t,y,trainedNet),[0 20],X0);
figure(1);
%M = acrobot_animate(t,y);
tau = 0*t;
for n=1:length(t)
    [dx,tau(n)] = acrobot_lstm(t(n),y(n,:)',trainedNet);
end
figure(2); subplot(211); plot(t,tau);
M = acrobot_animate(t,y);
