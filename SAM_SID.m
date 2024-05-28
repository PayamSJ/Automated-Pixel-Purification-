function [alpha,sid,ED]=SAM_SID(xi,y)
[m2,n2]=size(y);
for j=1:m2
yi=y(j,:);
    alpha(j)=acos(dot(xi,yi)/sqrt(dot(xi,xi)*dot(yi,yi)));
pk=yi/sum(yi);qk=xi/sum(xi);
Ip=-log10(pk);Iq=-log10(qk);
Dpq=sum(pk.*(Iq-Ip));
Dqp=sum(qk.*(Ip-Iq));
sid(j)=abs(Dpq+Dqp);
 ED(j)= sqrt(sum(dot(xi,yi)));
end
