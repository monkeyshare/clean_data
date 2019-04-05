接着在终端输入“sudo vim .bash_profile”回车，输入电脑密码，回车

按 “i” 进行输入

在最后面添加下面一段配置

“export PATH=/usr/local/bin:$PATH”

然后按 “ESC”退出编辑，输入“:wq”保存即可

wq前面有个冒号，不要看漏了！

最后在终端输入下面代码执行，使配置修改生效即可

“source .bash_profile”
