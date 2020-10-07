clear all
close all
clc

%
ks =900 ;% or 1040 Suspension Stiffness (N/m) 
kus = 2500;% Tire stiffness (N/m)
ms = 2.45;% or 2.5 Sprung Mass (kg) 
mus = 1;% Unsprung Mass (kg)
bsn = 7.5;% Suspension Inherent Damping coefficient (sec/m)
busn = 5;% Tire Inherent Damping coefficient (sec/m)

An = [ 0 1 0 0;-ks/ms -bsn/ms ks/ms bsn/ms;0 0 0 1;ks/mus bsn/mus -(ks+kus)/mus -(bsn+busn)/mus];
Bn = [0; 1/ms;  0; -1/mus];
Cn = [ 1 0 -1 0;-ks/ms -bsn/ms ks/ms bsn/ms];

Tstep = 0.001;
IC = [0 0 0 0]';
time = 6000;
state = zeros(time,4);
state(1,:) = IC;                                                                                                               
stateLQR = zeros(time,4);
stateLQR(1,:) = IC;
                                                                                                                                                                                                                                                                                                                             
statePID = zeros(time,4);
statePID(1,:) = IC;

oldPar = zeros(2,1);
phi = zeros(4,2);
u1 = zeros(6,1);
c = 1;
pOld = 1*eye(2)*10; %10000000
lmbda = 0.9; %0.90
Q = diag([450, 30, 5, 0.01]); %diag([5, 30, 5, 0.01]);
R = 0.01; %0.01
bsT = randi([4 9],1,1);
busT = randi([3 7],1,1);
parT = [bsT;busT];
par = parT + 0.01*randn(2,1);
zrdot = 0;
zr = 0.02;
zrold = 0;
xus = 0;
xusLQR = 0;
estPar = zeros(2,time);
KLQRn = lqr(An,Bn,Q,R);
Kp = 89.9882;
z = linspace(0,6000,6000)';
signal = -0.01*square(0.002*z,50)+0.01;
sig = [signal ;signal];
sig = sig(1000:7000-1);
%
for i = 1:time
    i
   zrplot(i) = zr;
if mod(i,2000) == 0 
    bsT = randi([4 9],1,1);
    busT = randi([3 7],1,1);
    parT = [bsT;busT];
end
zrold = zr;
zr = zr + (20*rand(1)-10)*0.0001;
%zr = sig(i);
zrdot = (zr-zrold)/Tstep;
estPar(:,i) = oldPar;


%if mod(i,1500) == 0
 %   zrold = zr;
    %a = randi([1,3]);
    %rangeZR = [0.02 0 0.02];
    %zr = randi([-6 6])/100;
    %zr = rangeZR(a);
    %zrdot = (zr-zrold)/Tstep;
    %zrdot = 0;
%    if zr == 0.02
%        zr = 0;
%        zrdot = 20;
%    elseif zr == 0
%        zr = 0.02;
%        zrdot = -20;
%    end
% 
%else
%    zrdot = 0;
%end


bsEst = oldPar(1,1);
busEst = oldPar(2,1);
bsValue(i,1) = par(1,1);
busValue(i,1) = par(2,1);
zrdotValue(i,:) = zrdot;
zrValue(i,:) = zr;

d = [0; 0; 0; busEst/mus*zrdot+kus/mus*zr];
A = [ 0 1 0 0;-ks/ms -bsEst/ms ks/ms bsEst/ms;0 0 0 1;ks/mus bsEst/mus -(ks+kus)/mus -(bsEst+busEst)/mus];
B = [0; 1/ms;  0; -1/mus];
C = [ 1 0 -1 0;-ks/ms -bsEst/ms ks/ms bsEst/ms];
D = [0;1/ms];

% LQR
refLQR = [xusLQR,0,zr,0];
xeLQR = stateLQR(i,:)' - refLQR';
uLQR(i,:) = -KLQRn*xeLQR;

k1 = eqn(stateLQR(i,:)', A, B,uLQR(i,:),d);
k2 = eqn(stateLQR(i,:)'+0.5*k1*Tstep, A, B,uLQR(i,:),d);
k3 = eqn(stateLQR(i,:)'+0.5*k2*Tstep, A, B,uLQR(i,:),d);
k4 = eqn(stateLQR(i,:)'+k3*Tstep, A, B,uLQR(i,:),d);
x_k1 = stateLQR(i,:)' + (k1 + 2*k2 + 2*k3 + k4)*Tstep/6;
xusLQR = x_k1(3);
stateLQR(i+1,:) = x_k1';
out2(i,:) = C*stateLQR(i,:)'+D*uLQR(i,:);

%xeLQRn = stateLQRn(i,:)' - refLQR';
%uLQRn(i,:) = -KLQRn*xeLQRn;
%k1 = eqn(stateLQRn(i,:)', A, B,uLQRn(i,:),d);
%k2 = eqn(stateLQRn(i,:)'+0.5*k1*Tstep, A, B,uLQRn(i,:),d);
%k3 = eqn(stateLQRn(i,:)'+0.5*k2*Tstep, A, B,uLQRn(i,:),d);
%k4 = eqn(stateLQRn(i,:)'+k3*Tstep, A, B,uLQRn(i,:),d);
%x_k1 = stateLQRn(i,:)' + (k1 + 2*k2 + 2*k3 + k4)*Tstep/6;
%stateLQRn(i+1,:) = x_k1';

% PID
xePID = statePID(i,2)';
uPID(i,:) = -Kp*xePID;
k1 = eqn(statePID(i,:)', An, Bn,uPID(i,:),d);
k2 = eqn(statePID(i,:)'+0.5*k1*Tstep, An, Bn,uPID(i,:),d);
k3 = eqn(statePID(i,:)'+0.5*k2*Tstep, An, Bn,uPID(i,:),d);
k4 = eqn(statePID(i,:)'+k3*Tstep, An, Bn,uPID(i,:),d);
x_k1 = statePID(i,:)' + (k1 + 2*k2 + 2*k3 + k4)*Tstep/6;
statePID(i+1,:) = x_k1';
out3(i,:) = Cn*statePID(i,:)'+D*uPID(i,:);

% DRL
ref = [xus,0,zr,0];
x = state(i,:)';
xe = x - ref';
xe2 = abs(xe(2,1));
u = actionNN(xe2);
u = 1*u*sign(xe(2,1));
%out(i,:) = Cn*x+D*u;
out(i,:) = Cn*x+D*u;

force(i) = u;
k1 = eqn(x, An, Bn,u,d);
k2 = eqn(x+0.5*k1*Tstep, An, Bn,u,d);
k3 = eqn(x+0.5*k2*Tstep, An, Bn,u,d);
k4 = eqn(x+k3*Tstep, An, Bn,u,d);
x_k = x + (k1 + 2*k2 + 2*k3 + k4)*Tstep/6;
state(i+1,:) = x_k';
xen = x_k - ref';
xs = x_k(1);
xus = x_k(3);
xsdot = x_k(2);
xusdot = x_k(4);

% Estimation
phi = [(xsdot-xusdot)/mus; -xusdot/mus];
par = parT + 0.01*randn(2,1);
%u1 = actionNN(xen(2,1));
output(i,:) = (phi'*par)';
y = output(i,:)';
K = pOld * phi * inv(lmbda+phi'*pOld*phi);
err = y-(phi'*oldPar);
EstErr(i,:) = err;
newPar = oldPar + K * err;
pNew = ((eye(2)-K*phi')*pOld)*(1/lmbda);
pOld = pNew;
oldPar = newPar;

bsError(i) = abs((estPar(1,i)-bsValue(i,1))/bsValue(i,1));
busError(i) = abs((estPar(2,i)-busValue(i,1))/busValue(i,1));

end
%
force = force';

%%
estPar = estPar(:,1000:6999);
statePID = statePID(1000:6999,:);
stateLQR = stateLQR(1000:6999,:);
bsValue = bsValue(1000:6999,:);
state = state(1000:6999,:);
busValue = busValue(1000:6999,:);
out = out(1000:6999,:);
out3 = out3(1000:6999,:);
out2 = out2(1000:6999,:);
force = force(1000:6999,:);
uPID = uPID(1000:6999,:);
uLQR = uLQR(1000:6999,:);

%%
acc = sum(abs(out(:,2)))/length(out(:,2))
acc2 = sum(abs(out3(:,2)))/length(out3(:,2))
accLQR = sum(abs(out2(:,2)))/length(out2(:,2))

force1 = sum(abs(force(:)))/length(force(:))
force2 = sum(abs(uPID(:)))/length(uPID(:))
forceLQR = sum(abs(uLQR(:)))/length(uLQR(:))

%%
close all
t = (1000:6999)*Tstep;
t = t-1;
j = 1;

%
for i = 1:length(t)
    i
    if mod(i-1,20) == 0
        state1(j,:) = state(i,:);
        %m1(j,:) = m(i,:);
        statePID1(j,:) = statePID(i,:);
        stateLQR1(j,:) = stateLQR(i,:);
        estPar1(j,:) = estPar(:,i);
        bsValue1(j,:) = bsValue(i,:);
        busValue1(j,:) = busValue(i,:);
        forceDRL(j,:) = force(i,:);
        forcePID(j,:) = uPID(i,:);
        forceLQR(j,:) = uLQR(i,:);
        zrplot1(j,:) = zrplot(:,i);
        outDRL(j,:) = out(i,:);
        outPID(j,:) = out3(i,:);
        outLQR(j,:) = out2(i,:);
        j = j +1;
    end   
    
end
t1 = (1:length(state1(:,1)))*0.02;
t2 = (1:length(zrplot))*0.1;

%%
figure(1)
plot(t1,estPar1(:,1),'-', 'LineWidth', 1.75)
hold on 
plot(t1, bsValue1(:,1),'--', 'LineWidth', 3)
ylabel('b_s')
xlabel('Time (s)')
legend('Est b_s','b_s')

figure(2)
plot(t1,estPar1(:,2),'-', 'LineWidth', 1.75)
hold on 
plot(t1, busValue1(:,1),'--', 'LineWidth',3)

ylabel('b_{us}')
xlabel('Time (s)')
legend('Est b_{us}','b_{us}')
%%
figure(3)
plot(t1,state1(:,1),'-', 'LineWidth', 2)
hold on
plot(t1,statePID1(:,1),'--', 'LineWidth', 3)
%hold on
%plot(t1,stateLQR1(:,1),':', 'LineWidth', 1.75)
ylabel('x_{s} (m)')
xlabel('Time (s)')
legend('DRL','PID')

figure(4)
plot(t1,state1(:,2),'-', 'LineWidth', 2)
hold on
plot(t1,statePID1(:,2),'--', 'LineWidth', 3)
%hold on
%plot(t1,stateLQR1(:,2),':', 'LineWidth', 1.75)
ylabel('x_{us} (m)')
xlabel('Time (s)')
legend('DRL','PID')

figure(5)
plot(t2,zrplot,'-', 'LineWidth', 2)
ylabel({'$z_{r} (m)$'},'Interpreter','latex','FontSize',15)
xlabel('Time (s)')
legend({'$z_{r}$'},'Interpreter','latex','FontSize',15)

figure(6)
plot(t2,zrdotValue,'-', 'LineWidth', 2)
ylabel({'$\dot{z}_{r} (m)$'},'Interpreter','latex','FontSize',15)
xlabel('Time (s)')
legend({'$\dot{z}_{r}$'},'Interpreter','latex','FontSize',15)

%%
figure(1)
plot(t,out(:,2),'-.', 'LineWidth', 1.75)
hold on
plot(t,out3(:,2),'-', 'LineWidth', 1.75)
ylabel('Body Acceleration $\dot{x}_{s}$','Interpreter','latex', 'FontSize', 15)
xlabel('Time in sec')
legend('DRL','PID')
%%
figure(2)
plot(t1,outDRL(:,2), '-', 'LineWidth', 2.5)
hold on
plot(t1,outPID(:,2), '--', 'LineWidth', 2.5)
%hold on
%plot(t1,outLQR(:,2), ':', 'LineWidth', 2)
ylabel({'$\ddot{x}_{s} (\frac{m}{s^{2}})$'},'Interpreter','latex','FontSize',15)
xlabel('Time (s)')
legend('DRL','PID')
%%
figure(3)
plot(t,abs(EstErr(:,1)), 'LineWidth', 1.75)
ylabel('b_{us} Error')
xlabel('Time in sec')
legend('Measured b_{us} - Estimated b_{us}')
%%

figure(1)
semilogy(t,bsError, 'LineWidth', 1.75)
ylabel('b_{s} Error')
xlabel('Time in sec')
%legend('Measured b_{s} - Estimated b_{s}')

figure(2)
semilogy(t,busError, 'LineWidth', 1.75)
ylabel('b_{us} Error')
xlabel('Time in sec')
%legend('Measured b_{us} - Estimated b_{us}')

figure(3)
semilogy(abs(EstErr(:,1)), 'LineWidth', 1.75)
ylabel('b_{us} Error')
xlabel('Time in sec')
legend('Measured b_{us} - Estimated b_{us}')

%%
t = (1:time)*Tstep;

figure(1)
plot(t,stateLQRn(1:time,1), 'LineWidth', 1.75)
hold on 
plot(t,stateLQR(1:time,1), 'LineWidth', 1.75)
ylabel('x_s')
xlabel('Time in sec')
legend('LQR with Nominal Values','LQR with Estimated Values')

%%
figure(1)
t = (1:time)*Tstep;
plot(t,zrdotValue, 'LineWidth', 1.75)
ylabel('Rate of Change of the Road Profile (m/s)')
xlabel('Time in seconds')
%%
figure(2)
t = (1:time+1)*Tstep;
plot(t,state(:,2), 'LineWidth', 1.75)
ylabel('$\dot{x_{s}}$','interpreter','latex','FontSize',18)
xlabel('Time in sec')
%%
figure(2)
t = (1:time)*Tstep;
plot(t,force, 'LineWidth', 1.75)
ylabel('Force in N','interpreter','latex','FontSize',15)
xlabel('Time in sec')
%%
xsdot = state(:,2);
save('xsdot.mat','xsdot');
%%
figure(2)
t = (1:time)*Tstep;
plot(t,zrplot, 'LineWidth', 1.75)
ylabel('Road Profile in m','interpreter','latex','FontSize',15)
xlabel('Time in sec')



