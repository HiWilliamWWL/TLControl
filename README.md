# TLControl
TL_Control: Trajectory and Language Control for Human Motion Synthesis

<div class="section abstract">
	<h2>Abstract</h2><br>
	<div class="row" style="margin-bottom:10px">
		<div class="col" style="text-align:center">
			<img class="thumbnail" src="assets/TLControl_Teaser.png" style="width:90%; margin-bottom:20px">
		</div>

</div>
	<p>
		Controllable human motion synthesis is essential for applications in AR/VR, gaming, movies, and embodied AI. Existing methods often focus solely on either language or full trajectory control, lacking precision in synthesizing motions aligned with user-specified trajectories, especially for multi-joint control. To address these issues, we present TLControl, a new method for realistic human motion synthesis, incorporating both low-level trajectory and high-level language semantics controls. Specifically, we first train a VQ-VAE to learn a compact latent motion space organized by body parts. We then propose a Masked Trajectories Transformer to make coarse initial predictions of full trajectories of joints based on the learned latent motion space, with user-specified partial trajectories and text descriptions as conditioning. Finally, we introduce an efficient test-time optimization to refine these coarse predictions for accurate trajectory control. Experiments demonstrate that TLControl outperforms the state-of-the-art in trajectory accuracy and time efficiency, making it practical for interactive and high-quality animation generation.			</p>
</div>