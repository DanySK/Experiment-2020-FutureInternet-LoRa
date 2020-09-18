@file:JvmName("CartesianProduct")
package it.unibo.gradle

import com.google.common.collect.Lists
import java.io.File
import java.util.*

fun main(args: Array<String>) {
    val pathDir = args[0]
    val pathFile = args[1]
    var config = ConfigurationReader(pathFile)
    val configNet = config.configurationNetworkInput
    val configBroker = config.brokerHostConfigInput
    val configHosts = config.hostsConfig

    val lists = listOf(
        configNet.seed.map { Pair("seed", it) },
        configNet.gamma.map { Pair("gamma", it) },
        configNet.beta.map { Pair("beta", it) },
        configNet.dee.map { Pair("dee", it) },
        configNet.dcc.map { Pair("dcc", it) },
        configNet.dec.map { Pair("dec", it) },
        configNet.dscnd.map { Pair("dscnd", it) },
        configBroker.type.map { Pair("broker", it) }
    )
    val cp = Lists.cartesianProduct(lists).map { it.toMap() }

    println("end cartesian product")
    println("Start write files at ${Calendar.getInstance().time}")

    cp.map {
        Pair(
            ConfigurationNetworkOutput(
                seed = it["seed"].toString().toLong(),
                gamma = it["gamma"].toString().toDouble(),
                beta = it["beta"].toString().toDouble(),
                dee = it["dee"].toString().toDouble(),
                dcc = it["dcc"].toString().toDouble(),
                dec = it["dec"].toString().toDouble(),
                dscnd = it["dscnd"].toString().toDouble(),
                dlocalhost = configNet.dlocalhost
            ),
            BrokerHostConfigOutput(
                type = it["broker"].toString(),
                dataRate = configBroker.dataRate,
                id = configBroker.id
            )
        )
    }.forEachIndexed { index, pair ->
        val file = File(pathDir, "config_$index.toml")
        ConfigurationWriter.writeToFile(file, pair.first, pair.second, configHosts)
    }

    println("end write files at ${Calendar.getInstance().time}")
}
