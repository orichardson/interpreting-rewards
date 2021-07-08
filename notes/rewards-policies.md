
<!-- # THOUGHTS -->
## Reward-Free Learning

You can play a lot of games by purely exploring. Why? Because you have the most freedom, and most interesting possibilities by playing the game in the way that was "intended" by the developer. Dying is boring, because you have to start over (or wait while you're dead). The idea is that your reward signal is taken to be your surprise upon learning the state $s'$ that you transition to, given the state $s$ you are in now. That is, if $\beta : S \times A \to \Delta S$ represents your transition belief, the reward of transition $\rho(s',a,s)$ to $s'$ is given by $- \log {\beta(s'\mid a,s)}$. If we fix the real transition dynamics $\tau: S\times A \to \Delta S$, then the dependence on $s'$ is artificial, and we may as well write
so that in expectation over the true transition probabilities giving us the following reward function of the usual type $\rho : S \times A \to \mathbb R$:
\[
  \rho(a,s) := \mathbb E_{s'\sim \tau(S'\mid a,s)} \log \frac1{\beta(s'\mid a,s)},
\]
also which take the form of a cross entropy, may be written as
\[ \rho(a,s) = D\mkern-14muD\Big(\tau|as ~\Vert~\beta|as\Big)
  + \mathop{\mathrm H}(\tau|as)\]
where, the distribution over $X$ given by conditiong the joint  distribution $\mu(X,A,S)$ on $A = a$ and $S = s$ is written as $\mu|as$.

As a function of the belief $\beta$, this formula corresponds to a supervised learning objective with cross-entropy; as such it's a useful objectie if our task is to alter $\beta$ to match $\tau$, in context $(a,s)$.

But here it functions as an objective for our policy $\pi : S \to \Delta A$.


**Death.** Adam, and the [curisotiy-driven learning paper I read][1], both advocate for finite time-horizons, because death can leak information about the reward signal (i.e., it is boring). But this is a feature, in my view.



[1]: https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf "Curisotiy-Driven Learning"

[oa-notes]: https://docs.google.com/document/d/1PCVUM4O0pdiDnrnGyI6iZvd_BWElc8X0_biCDqtSXu4
