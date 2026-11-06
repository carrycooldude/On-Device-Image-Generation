package com.example.on_device_image_generation

import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.doAfterTextChanged
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.example.on_device_image_generation.databinding.ActivityMainBinding
import com.example.on_device_image_generation.model.QaiHubModelType
import com.example.on_device_image_generation.ui.ImageGenViewModel
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: ImageGenViewModel by viewModels()
    private val models = QaiHubModelType.entries.toList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            models.map { it.displayName }
        )
        binding.spinnerModel.adapter = adapter
        binding.spinnerModel.setSelection(models.indexOf(viewModel.uiState.value.selectedModel))
        binding.spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                viewModel.selectModel(models[pos])
            }
            override fun onNothingSelected(p: AdapterView<*>?) = Unit
        }

        binding.etPrompt.setText(viewModel.uiState.value.prompt)
        binding.etPrompt.doAfterTextChanged { viewModel.updatePrompt(it?.toString().orEmpty()) }
        binding.btnGenerate.setOnClickListener { viewModel.startGeneration() }

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect { s ->
                    binding.btnGenerate.isEnabled = !s.isGenerating
                    binding.progressBar.visibility = if (s.isGenerating) View.VISIBLE else View.GONE
                    val p = s.progress
                    if (p != null && p.totalSteps > 0) {
                        binding.progressBar.progress = p.currentStep * 100 / p.totalSteps
                    }
                    binding.tvStatus.text = s.errorMessage ?: p?.stage ?: s.statusMessage.orEmpty()
                    s.lastResult?.let { binding.ivOutput.setImageBitmap(it.outputBitmap) }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshModelStatuses()
    }
}
