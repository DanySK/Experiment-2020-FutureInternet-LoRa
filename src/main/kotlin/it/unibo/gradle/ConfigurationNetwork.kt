package it.unibo.gradle

import com.uchuhimo.konf.Config
import com.uchuhimo.konf.ConfigSpec
import com.uchuhimo.konf.source.toml.toToml
import java.io.File

// region input model
data class ConfigurationNetworkInput(
    val seed: List<Long>,
    val gamma: List<Double>,
    val beta: List<Double>,
    val dee: List<Double>,
    val dcc: List<Double>,
    val dec: List<Double>,
    val dscnd: List<Double>,
    val dlocalhost: Double
) {
    companion object : ConfigSpec("") {
        val configurationNetwork by required<ConfigurationNetworkInput>()
    }
}

data class BrokerHostConfigInput(
    val type: List<String>,
    val dataRate: Double,
    val id: String? = null
) {
    companion object : ConfigSpec("") {
        val brokerHostConfig by required<BrokerHostConfigInput>()
    }
}
// endregion

// region input/output model
data class HostConfig(
    val type: String,
    val dataRate: Double? = null,
    val id: String? = null
) {
    companion object : ConfigSpec("") {
        val hostsConfig by required<List<HostConfig>>()
    }
}
// endregion

// region output model
data class ConfigurationNetworkOutput(
    val seed: Long,
    val gamma: Double,
    val beta: Double,
    val dee: Double,
    val dcc: Double,
    val dec: Double,
    val dscnd: Double,
    val dlocalhost: Double
) {
    companion object : ConfigSpec("") {
        val configurationNetwork by required<ConfigurationNetworkOutput>()
    }
}

data class BrokerHostConfigOutput(
    val type: String,
    val dataRate: Double,
    val id: String? = null
) {
    companion object : ConfigSpec("") {
        val brokerHostConfig by required<BrokerHostConfigOutput>()
    }
}
// endregion

class ConfigurationReader(path: String) {

    val configurationNetworkInput: ConfigurationNetworkInput
    val brokerHostConfigInput: BrokerHostConfigInput
    val hostsConfig: List<HostConfig>

    init {
        val config = Config {
            addSpec(ConfigurationNetworkInput)
            addSpec(BrokerHostConfigInput)
            addSpec(HostConfig)
        }.from().toml.file(path)

        configurationNetworkInput = config[ConfigurationNetworkInput.configurationNetwork]
        brokerHostConfigInput = config[BrokerHostConfigInput.brokerHostConfig]
        hostsConfig = config[HostConfig.hostsConfig]
    }
}

object ConfigurationWriter {

    fun writeToFile(
        path: File,
        configurationNetworkOutput: ConfigurationNetworkOutput,
        brokerHostConfigOutput: BrokerHostConfigOutput,
        hostsConfigOutput: List<HostConfig>
    ) {
        val config = Config {
            addSpec(ConfigurationNetworkOutput)
            addSpec(BrokerHostConfigOutput)
            addSpec(HostConfig)
        }

        config[ConfigurationNetworkOutput.configurationNetwork] = configurationNetworkOutput
        config[BrokerHostConfigOutput.brokerHostConfig] = brokerHostConfigOutput
        config[HostConfig.hostsConfig] = hostsConfigOutput

        config.toToml.toFile(path)
    }
}
