function Pix_purify=Purification(Xi)
Mean_end=mean(Xi);
[alpha,sid,ED]=SAM_SID(Mean_end,Xi);
X95_alpha= norminv([0.1 0.90],mean(alpha),std(alpha));
X95_ED= norminv([0.1 0.90],mean(ED),std(ED));
Idi=find(ED>X95_ED(1) & ED<X95_ED(2) & alpha>X95_alpha(1) & alpha<X95_alpha(2));
Pix_purify=Xi(idi);